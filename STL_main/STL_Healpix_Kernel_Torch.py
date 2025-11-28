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
class STL_Healpix_Kernel_Torch:
    """HealpixKernel_torch
    HEALPix analogue of STL2DKernel.

    Attributes
    ----------
    DT : str
        Data type identifier ("").
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
    def __init__(self, array, nside=None, cell_ids=None, nest=True):
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

        self.dg = 0

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
        
        #smooth kernel coded
        # x,y=np.meshgrid(np.arange(5)-2,np.arange(5)-2)
        # sigma=1.0
        # np.exp(-(x**2+y**2)/(2*sigma**2))
        self.smooth_kernel = torch.Tensor(np.array([0.01831564, 0.082085  , 0.13533528, 0.082085  , 0.01831564,
                            0.082085  , 0.36787944, 0.60653066, 0.36787944, 0.082085  ,
                            0.13533528, 0.60653066, 1.        , 0.60653066, 0.13533528,
                            0.082085  , 0.36787944, 0.60653066, 0.36787944, 0.082085  ,
                            0.01831564, 0.082085  , 0.13533528, 0.082085  , 0.01831564]).reshape(1,1,25))
        self.smooth_kernel = self.smooth_kernel/self.smooth_kernel.sum()
        
        # --- cache for SphericalStencil objects (one per resolution / kernel config) ---
        # key: (dg, kernel_sz, n_gauges, gauge_type, nest)
        self._stencil_cache = {}

    ###########################################################################
    def _get_stencil(self, kernel_sz: int, n_gauges: int = 1,
                     gauge_type: str = "cosmo"):
        """
        Return a cached SphericalStencil for the current resolution (dg, nside)
        and given kernel_sz / n_gauges / gauge_type.

        If it doesn't exist yet, build it once and store in self._stencil_cache.
        """
        # Key identifies geometry: dg controls nside and Npix, and nest is important
        key = (int(self.dg), int(kernel_sz), int(n_gauges), str(gauge_type), bool(self.nest))

        stencil = self._stencil_cache.get(key, None)
        if stencil is None:
            # Build it once
            cid_np = self.cell_ids.detach().cpu().numpy().astype(np.int64)
            stencil = SphericalStencil(
                nside=self.nside,
                kernel_sz=kernel_sz,
                nest=self.nest,
                cell_ids=cid_np,
                n_gauges=n_gauges,
                gauge_type=gauge_type,
                device=self.device,
                dtype=self.dtype,
            )
            self._stencil_cache[key] = stencil
        else:
            # Rebind device/dtype if they've changed (geometry stays valid)
            stencil.device = self.device
            stencil.dtype = self.dtype

        return stencil

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
        Copy a STL_Healpix_Kernel_Torch instance.

        Parameters
        ----------
        empty : bool
            If True, set array to None and cell_ids to None.

        Returns
        -------
        STL_Healpix_Kernel_Torch
            Shallow copy of metadata + (optionally) arrays.
        """
        new = object.__new__(STL_Healpix_Kernel_Torch)

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
        new.smooth_kernel = self.smooth_kernel
        new._stencil_cache = self._stencil_cache

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
    def modulus(self, inplace=False):
        """
        Compute the modulus (absolute value) of the data.

        For complex data, uses torch.abs. For real data, it is just |x|.
        """
        data = self.copy(empty=False) if not inplace else self

        if data.MR:
            data.array = [torch.abs(a) for a in data.array]
        else:
            data.array = torch.abs(data.array)

        data.dtype = data.array[0].dtype if data.MR else data.array.dtype
        return data

    ###########################################################################
    def mean(self, square=False):
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
    def cov(self, data2=None, remove_mean=False):
        """
        Compute covariance along the pixel axis between self and data2.

        Only supports MR == False.

        Parameters
        ----------
        data2 : STL_Healpix_Kernel_Torch or None
            If None, compute auto-covariance of self.
        remove_mean : bool
            If True, subtract the mean before multiplying.

        Returns
        -------
        torch.Tensor
            Covariance values over the last dimension.
        """
        if self.MR:
            raise ValueError("cov currently supports only MR == False.")

        x = self.array
        if data2 is None:
            y = x
        else:
            if not isinstance(data2, STL_Healpix_Kernel_Torch):
                raise TypeError("data2 must be a STL_Healpix_Kernel_Torch instance.")
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

        cov = (x_c * y_c.conj()).mean(dim=dim)
        return cov

    ###########################################################################
    def _smooth_inplace(self):
        """
        Smooth the data in-place using self.smooth_kernel and SphericalStencil.Convol_torch.
        Shape of the data is preserved.
        """
        kernel_sz = int(np.sqrt(self.smooth_kernel.shape[-1]))
        nside_in = self.nside
        cid_np = self.cell_ids.detach().cpu().numpy().astype(np.int64)

        x = self.array
        *leading, K = x.shape
        if leading:
            B = int(np.prod(leading))
        else:
            B = 1
        x_bc = x.reshape(B, 1, K)

        # --- Get or build a cached stencil for smoothing ---
        stencil = self._get_stencil(kernel_sz=kernel_sz, n_gauges=1, gauge_type='cosmo')

        # Smooth via convolution on the sphere: (B,1,K) -> (B,1,K)
        w_smooth = self.smooth_kernel.to(device=self.device, dtype=self.dtype)
        y_bc = stencil.Convol_torch(x_bc, w_smooth, cell_ids=cid_np, nside=nside_in)

        if not isinstance(y_bc, torch.Tensor):
            y_bc = torch.as_tensor(y_bc, device=self.device, dtype=self.dtype)

        y = y_bc.reshape(*leading, K)
        self.array = y

    ###########################################################################
    def downsample(self, dg_out: int, inplace: bool = True, smooth: bool = True):
        """
        Downsample the data to a coarser dg_out level in one step, using NESTED
        binning on HEALPix indices.

        Logic:
          - Optionally smooth the map at current resolution to remove small scales.
          - Group pixels by their parent index in NESTED scheme:
                parent_id = cell_id // 4**Δg
            where Δg = dg_out - dg.
          - For each parent pixel, average all children pixels' values.

        Only supports MR == False.

        Parameters
        ----------
        dg_out : int
            Target downgrading level (dg_out >= self.dg >= 0).
        inplace : bool
            If True, modify the current object.
            If False, return a new object and leave self unchanged.
        smooth : bool
            If True, apply smoothing before binning.

        Returns
        -------
        STL_Healpix_Kernel_Torch
            Data at the desired dg_out resolution.
        """
        if self.MR:
            raise ValueError("downsample only supports MR == False.")

        dg_out = int(dg_out)
        if dg_out < 0:
            raise ValueError("dg_out must be non-negative.")

        if dg_out < self.dg:
            raise ValueError("Requested dg_out < current dg; upsampling not supported.")

        # Trivial case
        if dg_out == self.dg:
            return self if inplace else self.copy(empty=False)

        # Work on a copy or in-place
        data = self if inplace else self.copy(empty=False)

        # 1) Smoothing step (optional)
        if smooth:
            data._smooth_inplace()

        # 2) Binning in NESTED scheme
        delta_g = dg_out - data.dg
        factor_pix = 4 ** delta_g  # 4^Δg children per parent in NESTED

        cid = data.cell_ids  # (K,)
        # Parent indices in NESTED
        parent_ids = cid // factor_pix  # (K,)

        # We want to average data.array[..., K] over these parent_ids
        x = data.array
        *leading, K = x.shape
        if leading:
            B = int(np.prod(leading))
        else:
            B = 1

        # Flatten leading dims into batch dimension: (B, K)
        x_flat = x.reshape(B, K)

        # Unique parent indices and inverse mapping
        parent_unique, inv = torch.unique(parent_ids, return_inverse=True)
        Kc = parent_unique.numel()  # number of coarse pixels

        # Scatter-add sums into coarse bins
        out = torch.zeros(B, Kc, device=x.device, dtype=x.dtype)
        idx = inv.unsqueeze(0).expand(B, -1)  # (B, K)
        out.scatter_add_(1, idx, x_flat)

        # Compute counts per coarse pixel (non-zero by construction)
        counts = torch.bincount(inv, minlength=Kc).to(x.dtype)  # (Kc,)
        out = out / counts.unsqueeze(0)  # broadcast over batch

        # Reshape back to original leading dims + coarse pixel axis
        y = out.reshape(*leading, Kc)

        # Update data object
        data.array = y
        data.cell_ids = parent_unique.to(device=data.device, dtype=torch.long)
        data.dg = dg_out
        # New nside from N0 and dg_out (conceptually underlying grid)
        data.nside = data.N0[0] // (2 ** dg_out)

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
                tmp = tmp.downsample(dg, inplace=True)
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
        """
        if L is None:
            L = 4
        if kernel_size is None:
            kernel_size = 5
        if J is None:
            J = int(np.log2(self.N0))

        # Reuse cached stencils for this resolution / kernel
        stencil = self._get_stencil(kernel_sz=kernel_size,
                                    n_gauges=L, gauge_type='cosmo')
        stencil_smooth = self._get_stencil(kernel_sz=kernel_size,
                                           n_gauges=1, gauge_type='cosmo')

        return WavelateOperatorHealpixKernel_torch(
            stencil=stencil,
            stencil_smooth=stencil_smooth,
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

    def __init__(self, stencil: SphericalStencil, stencil_smooth: SphericalStencil, kernel_size: int, L: int, J: int,
                 device='cuda', dtype=torch.float):
        self.stencil = stencil
        self.stencil_smooth = stencil_smooth
        self.KERNELSZ = kernel_size
        self.L = L
        self.J = J
        self.device = torch.device(device)
        self.dtype = dtype
        self.WType = "HealpixWavelet"

        # Build (1, L, P) kernel, where P=K^2
        kernel_2d = self._wavelet_kernel(kernel_size, L)  # (1, L, K, K)
        self.kernel = kernel_2d.reshape(1, 1, kernel_size * kernel_size)  # (Ci=1, Co=L, P)

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
        mother_kernel = torch.exp(-(xx**2 + yy**2) / (2*sigma**2))[None, :, :]

        # Orientations done in the gauges paradigm
        angles_proj = 0.5 * torch.pi * (xx[None, ...])

        kernel = torch.complex(
            torch.cos(angles_proj) * mother_kernel,
            torch.sin(angles_proj) * mother_kernel,
        )

        # Zero-mean and normalization per orientation
        kernel = kernel - torch.mean(kernel, dim=(1, 2), keepdim=True)
        kernel = kernel / torch.sum(kernel.abs(), dim=(1, 2), keepdim=True)

        return kernel.reshape(1, 1, kernel_size, kernel_size)

    def get_L(self):
        return self.L

    def apply(self, data: STL_Healpix_Kernel_Torch, j: int):
        """
        Apply the wavelet convolution to a STL_Healpix_Kernel_Torch instance.

        Parameters
        ----------
        data : STL_Healpix_Kernel_Torch
            Input Healpix data with array of shape [..., K] and cell_ids aligned.
            Must be at downgrading level dg == j.
        j : int
            Scale index. We simply check consistency with data.dg.

        Returns
        -------
        STL_Healpix_Kernel_Torch
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
        wr = torch.real(self.kernel).to(device=data.device, dtype=data.dtype)
        wi = torch.imag(self.kernel).to(device=data.device, dtype=data.dtype)

        # Use the same stencil but rebind device/dtype if needed
        if (self.stencil.device != data.device) or (self.stencil.dtype != data.dtype):
            # No heavy re-init: we just update device/dtype (geometry is cached in stencil)
            self.stencil.device = data.device
            self.stencil.dtype = data.dtype

        # Convolution on sphere -> (B, L, K)
        y_bc = torch.complex(self.stencil.Convol_torch(x_bc, wr, cell_ids=cid.detach().cpu().numpy()),
                            self.stencil.Convol_torch(x_bc, wi, cell_ids=cid.detach().cpu().numpy()))
        if not isinstance(y_bc, torch.Tensor):
            y_bc = torch.as_tensor(y_bc, device=data.device, dtype=data.dtype)

        _, L, K_out = y_bc.shape
        y = y_bc.reshape(*leading, L, K_out)  # [..., L, K]

        # Wrap into a new STL_Healpix_Kernel_Torch (same nside, same cell_ids, same dg)
        out = data.copy(empty=True)
        out.MR = False
        out.array = y
        out.cell_ids = cid.clone()
        out.dg = data.dg
        out.nside = data.nside
        out.N0 = data.N0
        out.list_dg = None
        return out
        
    def apply_smooth(self, data: STL_Healpix_Kernel_Torch, inplace: bool = True):
        """
        Smooth the data by convolving with a smooth kernel derived from the
        wavelet orientation 0. The data shape is preserved.

        Parameters
        ----------
        data : STL_Healpix_Kernel_Torch
            Input Healpix data with array of shape [..., K] and cell_ids aligned.
        copy : bool
            If True, return a new STL_Healpix_Kernel_Torch instance.
            If False, modify the input object in-place and return it.

        Returns
        -------
        STL_Healpix_Kernel_Torch
            Smoothed data object with same shape as input (no extra L dimension).
        """
        x = data.array  # [..., K]
        cid = data.cell_ids
        *leading, K = x.shape

        # Flatten leading dims into batch dimension: (B, Ci=1, K)
        if leading:
            B = int(np.prod(leading))
        else:
            B = 1
        x_bc = x.reshape(B, 1, K)

        # Smooth kernel (Ci=1, Co=1, P)
        w_smooth = self.kernel.abs().to(device=data.device, dtype=data.dtype)

        # Make sure stencil uses the right device/dtype
        if (self.stencil_smooth.device != data.device) or (self.stencil_smooth.dtype != data.dtype):
            self.stencil_smooth.device = data.device
            self.stencil_smooth.dtype = data.dtype

        # Convolution on sphere -> (B, 1, K)
        y_bc = self.stencil_smooth.Convol_torch(
            x_bc,
            w_smooth,
            cell_ids=cid.detach().cpu().numpy()
        )
        
        if not isinstance(y_bc, torch.Tensor):
            y_bc = torch.as_tensor(y_bc, device=data.device, dtype=data.dtype)

        y = y_bc.reshape(*leading, K)  # same shape as input x

        # Copy or in-place update
        out = data.copy(empty=True) if not inplace else data
        out.array = y
        # metadata stays identical (nside, N0, dg, cell_ids, ...)
        return out

