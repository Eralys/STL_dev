import torch
from torch import nn
from torch.optim import LBFGS

# Suppose:
# - STLDataClass is your data wrapper class (e.g. STL2DKernel)
# - st_op is an operator such that st_op.apply(DC).to_flatten()
#   returns a 1D tensor of scattering coefficients.
# - target is your reference map of shape (1,1,128,128)
#   and r is the corresponding scattering vector.


class ScatteringMatchModel(nn.Module):
    """
    Model that holds a learnable signal u and
    computes its scattering statistics via st_op.
    """
    def __init__(self, st_op, STLDataClass, init_shape, device=None, dtype=None):
        super().__init__()
        self.st_op = st_op           # scattering operator (already built)
        self.STLDataClass = STLDataClass  # data wrapper class

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if dtype is None:
            dtype = torch.float32

        # Initialize u as a learnable parameter
        # You can also initialize from target if you prefer
        self.u = nn.Parameter(torch.randn(init_shape, device=device, dtype=dtype))

    def forward(self):
        """
        Forward pass: compute scattering coefficients of u
        and return them as a flattened 1D tensor.
        """
        # Wrap u in your STL data class
        DC_u = self.STLDataClass(self.u)

        # Apply scattering operator
        st_u = self.st_op.apply(DC_u)

        # Flatten the scattering statistics
        # (adapt this if your method is named differently)
        s_flat_u = st_u.to_flatten()

        return s_flat_u

def optimize_scattering_LBFGS(
    target,
    STLDataClass,
    SO_class,
    max_iter=200,
    lr=1.0,
    history_size=50,
    verbose=True,
    print_every=10,
):
    """
    Run LBFGS optimization to find u such that its scattering
    coefficients match those of `target`.

    Parameters
    ----------
    target : torch.Tensor or np.ndarray
        Reference data of shape (1, 1, Nx, Ny).
    STLDataClass : class
        Your data wrapper class (e.g. STL2DKernel).
    SO_class : class or callable
        Scattering operator constructor, called like SO_class(DC_target).
    max_iter : int
        Number of outer LBFGS iterations (we loop manually).
    lr : float
        Learning rate for LBFGS.
    history_size : int
        LBFGS history size.
    verbose : bool
        If True, print loss during training.
    print_every : int
        Print every `print_every` outer iterations.

    Returns
    -------
    u_opt : torch.Tensor
        Optimized signal of shape (1,1,Nx,Ny).
    loss_history : list[float]
        List of loss values at each outer iteration.
    """

    # Ensure target is a torch tensor on some device/dtype
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target = torch.as_tensor(target, device=device, dtype=torch.float32)

    # Build STLDataClass for target
    DC_target = STLDataClass(target)

    # Build scattering operator based on target geometry
    st_op = SO_class(DC_target)

    # Compute target scattering vector r (no grad needed)
    with torch.no_grad():
        r = st_op.apply(DC_target).to_flatten()
    r = r.detach()  # make sure it does not require gradients

    # Initialize model with a learnable u
    model = ScatteringMatchModel(
        st_op=st_op,
        STLDataClass=STLDataClass,
        init_shape=target.shape,
        device=device,
        dtype=target.dtype,
    )

    # LBFGS optimizer on model.u
    optimizer = LBFGS(
        [model.u],
        lr=lr,
        history_size=history_size,
        line_search_fn="strong_wolfe"
    )

    loss_history = []

    # We loop manually over outer iterations; LBFGS will call closure internally
    for it in range(max_iter):
        def closure():
            optimizer.zero_grad()
            # Scattering coefficients of current u
            s_flat_u = model()
            # Quadratic loss between target and current scattering coefficients
            loss = ((s_flat_u - r) ** 2).sum()
            loss.backward()
            return loss

        loss = optimizer.step(closure)

        # Convert loss to float
        loss_val = loss.item() if isinstance(loss, torch.Tensor) else float(loss)
        loss_history.append(loss_val)

        if verbose and (it % print_every == 0 or it == max_iter - 1):
            print(f"[LBFGS] iter {it+1}/{max_iter}, loss = {loss_val:.6e}")

    # Extract optimized u
    u_opt = model.u.detach()

    return u_opt, loss_history
    
