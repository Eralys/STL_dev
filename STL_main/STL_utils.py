import torch


class Gaussianise:
    """
    Rank-based Gaussianisation with invertible mapping via interpolation.

    Method:
      - Fit on a reference tensor x_ref.
      - Build empirical CDF F_X from sorted(x_ref).
      - Forward: for any x, approximate u = F_X(x) by linear interpolation,
        then z = Phi^{-1}(u) (standard normal).
      - Inverse: for any z, u = Phi(z), then approximate x = F_X^{-1}(u)
        by linear interpolation between sorted samples.

    This is designed to be differentiable w.r.t. z in the inverse,
    and w.r.t. x in the forward, except for the non-differentiable
    dependence of the *sorting* (piecewise constant).
    """

    def __init__(self, x_ref: torch.Tensor, eps: float = 1e-6):
        """
        Parameters
        ----------
        x_ref : torch.Tensor
            Reference data used to define the mapping. Arbitrary shape.
        eps : float
            Small value used to clamp probabilities away from 0 and 1.
        """
        if not torch.is_tensor(x_ref):
            x_ref = torch.as_tensor(x_ref)

        if not x_ref.is_floating_point():
            x_ref = x_ref.float()

        self.device = x_ref.device
        self.dtype = x_ref.dtype
        self.eps = eps

        # Flatten reference data
        x_flat = x_ref.reshape(-1)

        # Sort reference values
        x_sorted, _ = torch.sort(x_flat)  # (N,)
        N = x_sorted.numel()

        # Empirical CDF positions: (r+0.5)/N for ranks r=0..N-1
        ranks = torch.arange(N, device=self.device, dtype=self.dtype)
        u_sorted = (ranks + 0.5) / float(N)  # (N,)

        # Store mapping: x_sorted <-> u_sorted
        self.x_sorted = x_sorted
        self.u_sorted = u_sorted

    # ----- basic normal CDF and inverse CDF using erf/erfinv -----

    def _phi_inv(self, u: torch.Tensor) -> torch.Tensor:
        """
        Inverse CDF of standard normal using erfinv.
        Φ^{-1}(u) = sqrt(2) * erfinv(2u - 1)
        """
        u = u.clamp(self.eps, 1.0 - self.eps)
        return torch.sqrt(torch.tensor(2.0, device=u.device, dtype=u.dtype)) * torch.erfinv(2.0 * u - 1.0)

    def _phi(self, z: torch.Tensor) -> torch.Tensor:
        """
        CDF of standard normal using erf.
        Φ(z) = 0.5 * [1 + erf(z / sqrt(2))]
        """
        return 0.5 * (1.0 + torch.erf(z / torch.sqrt(torch.tensor(2.0, device=z.device, dtype=z.dtype))))

    # ----- helper: 1D monotone interpolation -----

    def _interp_1d(self, xq: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        1D linear interpolation: given sorted x, corresponding y, and query xq,
        approximate yq = f(xq).

        - x: (N,) sorted in ascending order
        - y: (N,)
        - xq: arbitrary shape

        Returns yq with same shape as xq.

        This uses torch.searchsorted to find interval indices, and then
        linear interpolation between neighbors. Outside the range, it
        clamps to endpoints.
        """
        # Flatten queries
        xq_flat = xq.reshape(-1)  # (M,)

        # Indices of the right neighbor for each query
        idx_hi = torch.searchsorted(x, xq_flat, right=True)  # (M,)
        idx_hi = idx_hi.clamp(0, x.numel() - 1)
        idx_lo = (idx_hi - 1).clamp(0, x.numel() - 1)

        x_lo = x[idx_lo]
        x_hi = x[idx_hi]
        y_lo = y[idx_lo]
        y_hi = y[idx_hi]

        # Avoid division by zero when x_hi == x_lo (duplicate x)
        denom = (x_hi - x_lo)
        denom = torch.where(denom.abs() < 1e-12, torch.ones_like(denom), denom)

        t = (xq_flat - x_lo) / denom
        yq_flat = y_lo + t * (y_hi - y_lo)

        return yq_flat.reshape(xq.shape)

    # ----- forward: data -> Gaussian -----

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Gaussianise input tensor x based on the reference distribution.

        Steps:
          - For each value, approximate u = F_X(x) via interpolation
            in (x_sorted, u_sorted).
          - Then z = Φ^{-1}(u).

        Parameters
        ----------
        x : torch.Tensor
            Data tensor (any shape).

        Returns
        -------
        z : torch.Tensor
            Gaussianised tensor with same shape as x.
        """
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, device=self.device, dtype=self.dtype)
        else:
            x = x.to(device=self.device, dtype=self.dtype)

        # Approximate empirical CDF of x using reference x_sorted -> u_sorted
        u = self._interp_1d(x, self.x_sorted, self.u_sorted)

        # Map to Gaussian
        z = self._phi_inv(u)
        return z

    __call__ = forward  # so you can do g(x) directly

    # ----- inverse: Gaussian -> data -----

    def invert(self, z: torch.Tensor) -> torch.Tensor:
        """
        Map Gaussianised tensor z back to the original data space
        using the inverse empirical CDF.

        Steps:
          - u = Φ(z)
          - x = F_X^{-1}(u) via interpolation in (u_sorted, x_sorted).

        Parameters
        ----------
        z : torch.Tensor
            Gaussian-space tensor (any shape).

        Returns
        -------
        x_rec : torch.Tensor
            Reconstructed data tensor in the original amplitude space.
        """
        if not torch.is_tensor(z):
            z = torch.as_tensor(z, device=self.device, dtype=self.dtype)
        else:
            z = z.to(device=self.device, dtype=self.dtype)

        # Map to uniform using normal CDF
        u = self._phi(z)

        # Approximate inverse empirical CDF using u_sorted -> x_sorted
        x_rec = self._interp_1d(u, self.u_sorted, self.x_sorted)
        return x_rec
