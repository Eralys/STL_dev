#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HEALPix kernel-based data class for STL.

Analogue of STL2DKernel, but:
  - data live on HEALPix pixels (..., Npix)
  - convolutions and downsampling are performed with SphericalStencil.

Assumptions
-----------
- Data are real (or complex) PyTorch tensors.
- Last dimension is always the pixel axis.
- Pixel indexing is HEALPix (NESTED or RING, consistent with SphericalStencil.nest).
"""

import numpy as np
import torch
import torch.nn.functional as F

from SphericalStencil import SphericalStencil   # adapt if needed


###############################################################################
class STLHealpixKernel:
    """
    HEALPix analogue of STL2DKernel.

    Attributes
    ----------
    DT : str
        Data type identifier ("HealpixKernel_torch").
    MR : bool
        If True, stores a list of arrays at multiple resolutions.
    N0 : int
        Initial HEALPix nside at dg=0.
    dg : int or None
        downgrading level (nside = N0 // 2**dg) if MR == False.
    list_dg : list[int] or None
        List of downgrading levels if MR == True.
    array : torch.Tensor or list[torch.Tensor]
        Data array(s):
          - MR == False : tensor of shape (..., Npix)
          - MR == True  : list of tensors with same leading dims, different Npix.
    cell_ids : torch.LongTensor or list[torch.LongTensor]
        Pixel indices corresponding to the last axis of array.
    device : torch.device
        Default device for internal tensors.
    dtype : torch.dtype
        Default dtype for internal tensors.
    """

    ###########################################################################
    def __init__(self, array, nside=None, cell_ids=None, dg=None, nest=True):
        """
        Constructor for single-resolution Healpix data (MR == False).

        Parameters
        ----------
        array : np.ndarray or torch.Tensor
            Input data of shape (..., Npix).
        nside : int
            HEALPix resolution at dg=0.
        cell_ids : array-like or None
            HEALPix pixel indices for the last dimension.
            If None, assume full sky with standard ordering [0..Npix-1].
        dg : int or None
            Current downgrading level (default 0).
        nest : bool
            Whether pixel indexing is NESTED (must be consistent with SphericalStencil).
        """
        if isinstance(array, list):
            raise ValueError("Only single-resolution array is accepted at construction.")

        # Basic metadata
        self.DT = "HealpixKernel_torch"
        self.MR = False
        self.nest = bool(nest)

        if dg is None:
            self.dg = 0
        else:
            self.dg = int(dg)

        # Store N0 as the "reference" resolution at dg=0
        if nside is None:
            nside=int(np.sqrt(array.shape[-1]//12))
            
        self.N0 = [int(nside)]
        # Current nside is N0 // 2**dg
        self.nside = self.N0[0] // (2 ** self.dg)

        # Convert array to tensor and determine device/dtype
        self.array = self.to_array(array)
        self.device = self.array.device
        self.dtype = self.array.dtype

        # Last dimension = Npix
        Npix = self.array.shape[-1]

        # Cell ids
        if cell_ids is None:
            # Assume full-sky coverage [0..Npix-1]
            self.cell_ids = torch.arange(Npix, device=self.device, dtype=torch.long)
        else:
            self.cell_ids = self._to_cell_ids_tensor(cell_ids, Npix)

        # Multi-resolution attributes
        self.list_dg = None

    ###########################################################################
    @staticmethod
    def _default_device():
        """Return a default device (cuda if available, else cpu)."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ###########################################################################
    def _to_cell_ids_tensor(self, cell_ids, Npix_expected=None):
        """
        Convert any cell_ids-like (list/np/tensor) to a 1D LongTensor on self.device.
        Optionally check that its length matches Npix_expected.
        """
        if isinstance(cell_ids, torch.Tensor):
            cid = cell_ids.to(device=self.device, dtype=torch.long).view(-1)
        else:
            cid = torch.as_tensor(cell_ids, device=self.device, dtype=torch.long).view(-1)

        if (Npix_expected is not None) and (cid.numel() != Npix_expected):
            raise ValueError(
                f"cell_ids length {cid.numel()} does not match Npix={Npix_expected}."
            )
        return cid

    ###########################################################################
    def to_array(self, array):
        """
        Transform input array (NumPy or PyTorch) into a PyTorch tensor.

        Parameters
        ----------
        array : np.ndarray or torch.Tensor
            Input array to be converted (shape [..., Npix]).

        Returns
        -------
        torch.Tensor
            Converted PyTorch tensor on GPU if available, else CPU.
        """
        if array is None:
            return None
        elif isinstance(array, list):
            return array

        device = self._default_device()

        if isinstance(array, np.ndarray):
            return torch.from_numpy(array).to(device)
        elif isinstance(array, torch.Tensor):
            return array.to(device)
        else:
            raise TypeError(f"Unsupported array type: {type(array)}")

    ###########################################################################
    def copy(self, empty=False):
        """
        Copy a STLHealpixKernel instance.

        Parameters
        ----------
        empty : bool
            If True, set array to None and cell_ids to None.

        Returns
        -------
        STLHealpixKernel
            Shallow copy of metadata + (optionally) arrays.
        """
        new = object.__new__(STLHealpixKernel)

        # Copy metadata
        new.DT = self.DT
        new.MR = self.MR
        new.nest = self.nest
        new.N0 = self.N0
        new.nside = self.nside
        new.dg = self.dg
        new.list_dg = list(self.list_dg) if self.list_dg is not None else None
        new.device = self.device
        new.dtype = self.dtype

        # Copy data
        if empty:
            new.array = None
            new.cell_ids = None
        else:
            if self.MR:
                new.array = [
                    a.clone() if isinstance(a, torch.Tensor) else None
                    for a in self.array
                ]
                new.cell_ids = [
                    cid.clone() if isinstance(cid, torch.Tensor) else None
                    for cid in self.cell_ids
                ]
            else:
                new.array = self.array.clone() if isinstance(self.array, torch.Tensor) else None
                new.cell_ids = self.cell_ids.clone() if isinstance(self.cell_ids, torch.Tensor) else None

        return new

    ###########################################################################
    def __getitem__(self, key):
        """
        Slice directly into the stored array(s). Only slices over leading dims;
        pixel axis is always kept.
        """
        new = self.copy(empty=True)

        if self.MR:
            if not isinstance(self.array, list):
                raise ValueError("MR=True but array is not a list.")

            # Slice each resolution with the same key
            new.array = [a[key] for a in self.array]
            new.cell_ids = self.cell_ids[:]  # same pixel ids per resolution
        else:
            new.array = self.array[key]
            new.cell_ids = self.cell_ids

        return new

    ###########################################################################
    def modulus_func(self, copy=False):
        """
        Compute the modulus (absolute value) of the data.

        For complex data, uses torch.abs. For real data, it is just |x|.
        """
        data = self.copy(empty=False) if copy else self

        if data.MR:
            data.array = [torch.abs(a) for a in data.array]
        else:
            data.array = torch.abs(data.array)

        data.dtype = data.array[0].dtype if data.MR else data.array.dtype
        return data

    ###########################################################################
    def mean_func(self, square=False):
        """
        Compute the mean over the last (pixel) dimension.

        Parameters
        ----------
        square : bool
            If True, use |x|^2 instead of x.

        Returns
        -------
        torch.Tensor
            Mean values over pixels. If MR=True, the last dimension is len(list_dg).
        """
        if self.MR:
            means = []
            for arr in self.array:
                arr_use = torch.abs(arr) ** 2 if square else arr
                means.append(arr_use.mean(dim=-1))
            # Stack along a new last dimension corresponding to list_dg
            return torch.stack(means, dim=-1)
        else:
            if self.array is None:
                raise ValueError("No data stored in this object.")
            arr_use = torch.abs(self.array) ** 2 if square else self.array
            return arr_use.mean(dim=-1)

    ###########################################################################
    def cov_func(self, data2=None, remove_mean=False):
        """
        Compute covariance along the pixel axis between self and data2.

        Only supports MR == False.

        Parameters
        ----------
        data2 : STLHealpixKernel or None
            If None, compute auto-covariance of self.
        remove_mean : bool
            If True, subtract the mean before multiplying.

        Returns
        -------
        torch.Tensor
            Covariance values over the last dimension.
        """
        if self.MR:
            raise ValueError("cov_func currently supports only MR == False.")

        x = self.array
        if data2 is None:
            y = x
        else:
            if not isinstance(data2, STLHealpixKernel):
                raise TypeError("data2 must be a STLHealpixKernel instance.")
            if data2.MR:
                raise ValueError("data2 must have MR == False.")
            if data2.dg != self.dg:
                raise ValueError("data2 must have the same dg as self.")
            y = data2.array

        dim = -1  # pixel axis

        if remove_mean:
            mx = x.mean(dim=dim, keepdim=True)
            my = y.mean(dim=dim, keepdim=True)
            x_c = x - mx
            y_c = y - my
        else:
            x_c = x
            y_c = y

        cov = (x_c * y_c).mean(dim=dim)
        return cov

    ###########################################################################
    def _downsample_once(self, kernel_sz=3, max_poll=False):
        """
        Downsample by one step in nside using SphericalStencil.Down.

        This is a single-level helper (dg -> dg+1).

        Returns
        -------
        new_array : torch.Tensor
            Downsampled data with pixel axis length K_out.
        new_cell_ids : torch.LongTensor
            Corresponding HEALPix pixel indices at the coarse resolution.
        new_nside : int
            New nside (typically nside // 2).
        """
        # Current geometry
        nside_in = self.nside
        cid_np = self.cell_ids.detach().cpu().numpy().astype(np.int64)

        # Prepare input as (B, Ci, K)
        x = self.array
        *leading, K = x.shape
        if leading:
            B = int(np.prod(leading))
        else:
            B = 1

        x_bc = x.reshape(B, 1, K)

        # Build SphericalStencil with the current nside
        stencil = SphericalStencil(
            nside=nside_in,
            kernel_sz=kernel_sz,
            nest=self.nest,
            cell_ids=cid_np,
            device=self.device,
            dtype=self.dtype,
        )

        # Call Down: expect (B,1,K_out), ids_out
        dim, cid_out = stencil.Down(x_bc, cell_ids=cid_np, nside=nside_in, max_poll=max_poll)

        # dim: Torch tensor or maybe numpy; convert to torch if needed
        if not isinstance(dim, torch.Tensor):
            dim = torch.as_tensor(dim, device=self.device, dtype=self.dtype)
        if isinstance(cid_out, torch.Tensor):
            cid_out_t = cid_out.to(device=self.device, dtype=torch.long).view(-1)
        else:
            cid_out_t = torch.as_tensor(cid_out, device=self.device, dtype=torch.long).view(-1)

        # New map shape (..., K_out)
        _, _, K_out = dim.shape
        new_array = dim.reshape(*leading, K_out)

        # Guess new nside (assuming factor 4 in Npix -> factor 2 in nside)
        Npix_in = 12 * nside_in ** 2
        Npix_out = cid_out_t.numel()
        # Protect against division by zero / weird cases
        if Npix_out > 0:
            nside_out = int(np.sqrt(Npix_out / 12.0) + 0.5)
        else:
            nside_out = max(nside_in // 2, 1)

        return new_array.to(self.device), cid_out_t, nside_out

    ###########################################################################
    def downsample(self, dg_out, copy=False):
        """
        Downsample the data to a coarser dg_out level using Healpix ud_grade_2.

        The logic is:
          dg_out >= dg
          nside_out = N0 // 2**dg_out

        Only supports MR == False.

        Parameters
        ----------
        dg_out : int
            Target downgrading level.
        copy : bool
            If True, return a new object, else modify in-place.

        Returns
        -------
        STLHealpixKernel
            Data at the desired dg_out resolution.
        """
        if self.MR:
            raise ValueError("downsample only supports MR == False.")
        dg_out = int(dg_out)
        if dg_out < 0:
            raise ValueError("dg_out must be non-negative.")
        if dg_out == self.dg and not copy:
            return self
        if dg_out < self.dg:
            raise ValueError("Requested dg_out < current dg; upsampling not supported.")

        data = self.copy(empty=False) if copy else self

        # Number of single steps we need
        dg_inc = dg_out - data.dg
        for _ in range(dg_inc):
            new_arr, new_cid, new_nside = data._downsample_once(kernel_sz=3, max_poll=False)
            data.array = new_arr
            data.cell_ids = new_cid
            data.nside = new_nside
            data.dg += 1

        return data

    ###########################################################################
    def downsample_toMR(self, dg_max):
        """
        Build a multi-resolution representation by downsampling level by level.

        The object returned has:
          - MR == True
          - list_dg = [0, 1, ..., dg_max]
          - array = [a_0, a_1, ..., a_dgmax]
          - cell_ids = [cid_0, cid_1, ..., cid_dgmax]
        """
        if self.MR:
            raise ValueError("downsample_toMR expects MR == False.")
        if self.dg != 0:
            raise ValueError("downsample_toMR assumes current data is at dg=0.")
        if dg_max < 0:
            raise ValueError("dg_max must be non-negative.")
        if self.array is None:
            raise ValueError("No data stored in this object.")

        list_arrays = []
        list_cids = []
        list_dg = []

        # Start from dg=0
        tmp = self.copy(empty=False)

        for dg in range(dg_max + 1):
            if dg > 0:
                tmp = tmp.downsample(dg, copy=True)
            list_arrays.append(tmp.array)
            list_cids.append(tmp.cell_ids)
            list_dg.append(dg)

        data = self.copy(empty=True)
        data.MR = True
        data.array = list_arrays
        data.cell_ids = list_cids
        data.list_dg = list_dg
        data.dg = None
        return data

    ###########################################################################
    def get_wavelet_op(self, kernel_size=None, L=None, J=None):
        """
        Build a Healpix wavelet operator, analogous to get_wavelet_op() in STL2DKernel.

        Parameters
        ----------
        kernel_size : int or None
            Tangent-plane stencil size (odd), default 5.
        L : int or None
            Number of orientations, default 4.
        J : int or None
            Number of scales. If None, set from N0 (roughly log2(N0)-2).

        Returns
        -------
        WavelateOperatorHealpixKernel_torch
        """
        if L is None:
            L = 4
        if kernel_size is None:
            kernel_size = 5
        if J is None:
            J = int(np.log2(self.N0)) - 2

        # We build a SphericalStencil at the current nside & cell_ids
        stencil = SphericalStencil(
            nside=self.nside,
            kernel_sz=kernel_size,
            nest=self.nest,
            cell_ids=self.cell_ids.detach().cpu().numpy(),
            device=self.device,
            dtype=self.dtype,
        )
        return WavelateOperatorHealpixKernel_torch(
            stencil=stencil,
            kernel_size=kernel_size,
            L=L,
            J=J,
            device=self.device,
            dtype=self.dtype,
        )


###############################################################################
class WavelateOperatorHealpixKernel_torch:
    """
    Healpix wavelet operator using SphericalStencil.

    - Build a directional wavelet kernel on a local KxK stencil (tangent plane).
    - Flatten to shape (Ci=1, L, P=K^2).
    - Use SphericalStencil.Convol_torch to convolve maps.

    For now we implement a simple "Morlet-like" directional kernel similar
    to WavelateOperator2Dkernel_torch in STL2DKernel.
    """

    def __init__(self, stencil: SphericalStencil, kernel_size: int, L: int, J: int,
                 device='cuda', dtype=torch.float):
        self.stencil = stencil
        self.KERNELSZ = kernel_size
        self.L = L
        self.J = J
        self.device = torch.device(device)
        self.dtype = dtype
        self.WType = "HealpixWavelet"

        # Build (1, L, P) kernel, where P=K^2
        kernel_2d = self._wavelet_kernel(kernel_size, L)  # (1, L, K, K)
        self.kernel = kernel_2d.reshape(1, L, kernel_size * kernel_size)  # (Ci=1, Co=L, P)

    def _wavelet_kernel(self, kernel_size: int, n_orientation: int, sigma=1.0):
        """
        Create a 2D directional wavelet kernel on a KxK grid, similar to
        WavelateOperator2Dkernel_torch._wavelet_kernel.

        Returns
        -------
        kernel : torch.Tensor
            Complex tensor of shape (1, n_orientation, K, K).
        """
        coords = torch.arange(
            kernel_size,
            device=self.device,
            dtype=self.dtype
        ) - (kernel_size - 1) / 2.0
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")

        # Isotropic Gaussian envelope
        mother_kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))[None, :, :]

        # Orientations
        angles = torch.arange(
            n_orientation, device=self.device, dtype=self.dtype
        ) / n_orientation * np.pi
        angles_proj = torch.pi * (
            xx[None, ...] * torch.cos(angles[:, None, None]) +
            yy[None, ...] * torch.sin(angles[:, None, None])
        )

        kernel = torch.complex(
            torch.cos(angles_proj) * mother_kernel,
            torch.sin(angles_proj) * mother_kernel,
        )

        # Zero-mean and normalization per orientation
        kernel = kernel - torch.mean(kernel, dim=(1, 2), keepdim=True)
        kernel = kernel / torch.sum(kernel.abs(), dim=(1, 2), keepdim=True)

        return kernel.reshape(1, n_orientation, kernel_size, kernel_size)

    def get_L(self):
        return self.L

    def apply(self, data: STLHealpixKernel, j: int):
        """
        Apply the wavelet convolution to a STLHealpixKernel instance.

        Parameters
        ----------
        data : STLHealpixKernel
            Input Healpix data with array of shape [..., K] and cell_ids aligned.
            Must be at downgrading level dg == j.
        j : int
            Scale index. We simply check consistency with data.dg.

        Returns
        -------
        STLHealpixKernel
            New object with array shape [..., L, K], same nside & cell_ids.
        """
        if j != data.dg:
            raise ValueError("j is not equal to data.dg; convolution not consistent with scale.")

        x = data.array  # [..., K]
        cid = data.cell_ids
        *leading, K = x.shape

        # Flatten leading dims into batch dimension: (B, Ci=1, K)
        if leading:
            B = int(np.prod(leading))
        else:
            B = 1
        x_bc = x.reshape(B, 1, K)

        # Kernel for SphericalStencil: (Ci=1, Co=L, P)
        ww = self.kernel.to(device=data.device, dtype=data.dtype)

        # Use the same stencil but rebind device/dtype if needed
        if (self.stencil.device != data.device) or (self.stencil.dtype != data.dtype):
            # No heavy re-init: we just update device/dtype (geometry is cached in stencil)
            self.stencil.device = data.device
            self.stencil.dtype = data.dtype

        # Convolution on sphere -> (B, L, K)
        y_bc = self.stencil.Convol_torch(x_bc, ww, cell_ids=cid.detach().cpu().numpy())
        if not isinstance(y_bc, torch.Tensor):
            y_bc = torch.as_tensor(y_bc, device=data.device, dtype=data.dtype)

        _, L, K_out = y_bc.shape
        y = y_bc.reshape(*leading, L, K_out)  # [..., L, K]

        # Wrap into a new STLHealpixKernel (same nside, same cell_ids, same dg)
        out = data.copy(empty=True)
        out.MR = False
        out.array = y
        out.cell_ids = cid.clone()
        out.dg = data.dg
        out.nside = data.nside
        out.N0 = data.N0
        out.list_dg = None
        return out
