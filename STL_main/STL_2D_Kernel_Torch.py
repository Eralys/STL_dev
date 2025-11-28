#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tuesday Nov 26 2025

Example methods for a test data type.

2D planar maps with convolution using kernel.

This class makes all computations in torch.

Characteristics:
    - in pytorch
    - assume real maps 
    - N0 gives x and y sizes for array shaped (..., Nx, Ny).
    - masks are supported in convolutions
"""
import math
import numpy as np
import torch
import torch.nn.functional as F

def _conv2d_circular(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    Backend-style 2D convolution mirroring FoCUS/BkTorch strategy.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape [..., Nx, Ny].
    w : torch.Tensor
        Kernel tensor of shape [O_c, wx, wy].

    Returns
    -------
    torch.Tensor
        Convolved tensor with shape [..., O_c, Nx, Ny].
    """

    *leading_dims, Nx, Ny = x.shape
    O_c, wx, wy = w.shape

    B = int(torch.prod(torch.tensor(leading_dims))) if leading_dims else 1
    x4d = x.reshape(B, 1, Nx, Ny)

    weight = w[:, None, :, :]
    pad_x = wx // 2
    pad_y = wy // 2

    x_padded = F.pad(x4d, (pad_y, pad_y, pad_x, pad_x), mode="circular")
    y = F.conv2d(x_padded, weight)

    return y.reshape(*leading_dims, O_c, Nx, Ny)


def _complex_conv2d_circular(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Complex-aware wrapper around ``_conv2d_circular``."""

    xr = torch.real(x) if torch.is_complex(x) else x
    xi = torch.imag(x) if torch.is_complex(x) else torch.zeros_like(xr)

    wr = torch.real(w) if torch.is_complex(w) else w
    wi = torch.imag(w) if torch.is_complex(w) else torch.zeros_like(wr)

    real_part = _conv2d_circular(xr, wr) - _conv2d_circular(xi, wi)
    imag_part = _conv2d_circular(xr, wi) + _conv2d_circular(xi, wr)

    if torch.is_complex(x) or torch.is_complex(w):
        return torch.complex(real_part, imag_part)
    return real_part

###############################################################################
###############################################################################
class STL_2D_Kernel_Torch:
    '''
    Class which contain the different types of data used in STL.
    Store important parameters, such as DT, N0, and the Fourier type.
    Also allow to convert from numpy to pytorch (or other type).
    Allow to transfer internally these parameters.
    
    Has different standard functions as methods (
    modulus, mean, cov, downsample)
    
    The initial resolution N0 is fixed, but the maps can be downgraded. The 
    downgrading factor is the power of 2 that is used. A map of initial 
    resolution N0=256 and with dg = 3 is thus at resolution 256/2^3 = 32.
    The downgraded resolutions are called N0, N1, N2, ...
    
    Can store array at a given downgradind dg:
        - attribute MR is False
        - attribute N0 gives the initial resolution
        - attribute dg gives the downgrading level
        - attribute list_dg is None
        - array is an array of size (..., N) with N = N0 // 2^dg 
    Or at multi-resolution (MR):
        - attribute MR is True
        - attribute N0 gives the initial resolution
        - attribute dg is None
        - attribute list_dg is the list of downgrading
        - array is a list of array of sizes (..., N1), (..., N2), etc., 
        with the same dimensions excepts N.
     
    Method usages if MR=True.
        - mean, cov give a single vector or last dim len(list_N)
        - downsample gives an output of size (..., len(list_N), Nout). Only 
          possible if all resolution are downsampled this way.
          
    The class initialization is the frontend one, which can work from DT and 
    data only. It enforces MR=False and dg=0. Two backend init functions for 
    MR=False and MR=True also exist.
    
    Attributes
    ----------
    - DT : str
        Type of data (1d, 2d planar, HealPix, 3d)
    - MR: bool
        True if store a list of array in a multi-resolution framework
    - N0 : tuple of int
        Initial size of array (can be multiple dimensions)
    - dg : int
        2^dg is the downgrading level w.r.t. N0. None if MR==False  
    - list_dg : list of int
        list of dowgrading level w.r.t. N0, None if MR==False
    - array : array (..., N) if MR==False
          liste of (..., N1), (..., N2), etc. if MR==True
          array(s) to store
         
    '''
    
    ###########################################################################
    def __init__(self, array, smooth_kernel=None,dg=None,N0=None):
        '''
        Constructor, see details above. Frontend version, which assume the 
        array is at N0 resolution with dg=0, with MR=False.
        
        More sophisticated Back-end constructors (_init_SR and _init_MR) exist.
        
        '''
        
        # Check that MR==False array is given
        if isinstance(array, list):
            raise ValueError("Only single resolution array are accepted.")
        
        # Main 
        self.DT = 'Planar2D_kernel_torch'
        self.MR = False
        if dg is None:
            self.dg = 0
            self.N0 = array.shape[-2:]
        else:
            self.dg=dg
            if N0 is None:
                raise ValueError("dg is given, N0 should not be None")
            self.N0=N0
        
        
        self.array = self.to_array(array)
        
        self.list_dg = None
        
        # Find N0 value
        self.device=self.array.device
        self.dtype=self.array.dtype

        if smooth_kernel is None:
            smooth_kernel=self._smooth_kernel(3)
        self.smooth_kernel=smooth_kernel
        
        
    def _smooth_kernel(self,kernel_size: int):
        """Create a 2D Gaussian kernel."""
        sigma=1
        coords = torch.arange(kernel_size, device=self.device, dtype=self.dtype) - (kernel_size - 1) / 2.0
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        return kernel.view(1, kernel_size, kernel_size)
        
            
    ###########################################################################
    def to_array(self,array):
        """
        Transform input array (NumPy or PyTorch) into a PyTorch tensor.
        Should return None if None.

        Parameters
        ----------
        array : np.ndarray or torch.Tensor
            Input array to be converted.

        Returns
        -------
        torch.Tensor
            Converted PyTorch tensor.
        """
        
        if array is None:
            return None
        elif isinstance(array, list):
            return array

        # Choose device: use GPU if available, otherwise CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if isinstance(array, np.ndarray):
            return torch.from_numpy(array).to(device)
        elif isinstance(array, torch.Tensor):
            return array.to(device)
        else:
            raise TypeError(f"Unsupported array type: {type(array)}")
            
    ###########################################################################
    def copy(self, empty=False):
        """
        Copy a STL_2D_Kernel_Torch instance.
        Array is put to None if empty==True.
        
        Parameters
        ----------
        - empty : bool
            If True, set array to None.
                    
        Output 
        ----------
        - STL_2D_Kernel_Torch
           copy of self
        """
        new = object.__new__(STL_2D_Kernel_Torch)

        # Copy metadata
        new.MR = self.MR
        new.N0 = self.N0
        new.dg = self.dg
        new.list_dg = list(self.list_dg) if self.list_dg is not None else None
        new.device = self.device
        new.dtype = self.dtype

        # Copy kernels
        new.smooth_kernel = (self.smooth_kernel.clone()
                             if isinstance(self.smooth_kernel, torch.Tensor)
                             else None)

        # Copy array
        if empty:
            new.array = None
        else:
            if self.MR:
                new.array = [a.clone() if isinstance(a, torch.Tensor) else None
                             for a in self.array]
            else:
                new.array = (self.array.clone()
                             if isinstance(self.array, torch.Tensor) else None)

        return new

    ###########################################################################
    def __getitem__(self, key):
        """
        To slice directly the array attribute. Produce a view of array, to 
        match with usual practices, allowing to conveniently pass only part
        of an instance.
        """
        new = self.copy(empty=True)

        if self.MR:
            if not isinstance(self.array, list):
                raise ValueError("MR=True but array is not a list.")

            if isinstance(key, (int, slice)):
                new.array = self.array[key]
                new.list_dg = self.list_dg[key] if self.list_dg is not None else None

                # If a single element is selected, keep MR=True with a single resolution
                if isinstance(key, int):
                    new.array = [new.array]
                    new.list_dg = [new.list_dg]
            else:
                raise TypeError("Indexing MR=True data only supports int or slice.")
        else:
            new.array = self.array[key]

        return new
  
    @staticmethod
    def _downsample_tensor(x: torch.Tensor, dg_inc: int) -> torch.Tensor:
        """
        Downsample a tensor by a factor 2**dg_inc along the last two
        dimensions using 2x2 mean pooling (FoCUS ud_grade strategy).

        Requires that both spatial dimensions be divisible by 2**dg_inc.
        """
        if dg_inc < 0:
            raise ValueError("dg_inc must be non-negative")
        if dg_inc == 0:
            return x

        scale = 2 ** dg_inc
        H, W = x.shape[-2:]
        if H % scale != 0 or W % scale != 0:
            raise ValueError(
                f"Cannot downsample from ({H},{W}) by 2^{dg_inc}: "
                "dimensions must be divisible."
            )

    # Create Gaussian kernel for anti-aliasing
        def get_gaussian_kernel(sigma=1.0, dtype=torch.float32):
            """Create a 2D Gaussian kernel with kernel_size based on sigma"""
            kernel_size = int(6 * sigma + 1)
            if kernel_size % 2 == 0:
                kernel_size += 1
            
            x_coord = torch.arange(kernel_size, dtype=dtype) - kernel_size // 2
            gauss_1d = torch.exp(-x_coord**2 / (2 * sigma**2))
            gauss_1d = gauss_1d / gauss_1d.sum()
            gauss_2d = gauss_1d.unsqueeze(0) * gauss_1d.unsqueeze(1)
            return gauss_2d.unsqueeze(0).unsqueeze(0), kernel_size

        leading_dims = x.shape[:-2]
        B = int(torch.prod(torch.tensor(leading_dims))) if leading_dims else 1
        y = x.reshape(B, 1, H, W)

        for _ in range(dg_inc):
            h, w = y.shape[-2:]
            if h % 2 != 0 or w % 2 != 0:
                raise ValueError("Downsampling requires even spatial dimensions at each step.")
            
            sigma = 1.0
            kernel, kernel_size = get_gaussian_kernel(sigma, dtype=y.dtype)
            kernel = kernel.to(y.device)
            padding = kernel_size // 2
            
            y = F.conv2d(y, kernel, padding=padding)
            y = F.avg_pool2d(y, kernel_size=2, stride=2)

        H2, W2 = y.shape[-2:]
        return y.reshape(*leading_dims, H2, W2)
  
    ###########################################################################
    def downsample_toMR_Mask(self, dg_max):
        '''
        Take a mask given at a dg=0 resolution, and put it at all resolutions
        from dg=0 to dg=dg_max, in a MR=True object.

        Each resolution is normalized to have unit mean (over spatial dims).
        '''
        if self.MR:
            raise ValueError("downsample_toMR_Mask expects MR == False.")
        if self.dg != 0:
            raise ValueError("Mask should be at dg=0 to build a multi-resolution mask.")
        if self.array is None:
            raise ValueError("No array stored in this object.")

        list_masks = []
        list_dg = list(range(dg_max + 1))

        for dg in list_dg:
            if dg == 0:
                m = self.array
            else:
                m = self._downsample_tensor(self.array, dg)

            if (m < 0).any():
                raise ValueError("Mask contains negative values; expected non-negative weights.")

            mean = m.mean(dim=(-2, -1), keepdim=True)
            m = m / mean.clamp_min(1e-12)
            list_masks.append(m)

        Mask_MR = self.copy(empty=True)
        Mask_MR.MR = True
        Mask_MR.dg = None
        Mask_MR.list_dg = list_dg
        Mask_MR.array = list_masks

        return Mask_MR

    ###########################################################################
    def _get_mask_at_dg(self, mask_MR, dg):
        """Helper to pick the mask at a given dg from a MR mask object."""
        if mask_MR is None:
            return None
        if not mask_MR.MR:
            raise ValueError("mask_MR must have MR=True.")
        if mask_MR.list_dg is None:
            raise ValueError("mask_MR.list_dg is None.")
        try:
            idx = mask_MR.list_dg.index(dg)
        except ValueError:
            raise ValueError(f"Mask does not contain dg={dg}.")
        return mask_MR.array[idx]

    ###########################################################################
    def downsample(self, dg_out, mask_MR=None, inplace=True):
        """
        Downsample the data to the dg_out resolution.
        Only supports MR == False.

        Downsampling is done in real space by average pooling, with factor
        2^(dg_out - dg) on both spatial axes.
        """
        if self.MR:
            raise ValueError("downsample only supports MR == False.")
        if dg_out < 0:
            raise ValueError("dg_out must be non-negative.")
        if dg_out == self.dg and not copy:
            return self
        if dg_out < self.dg:
            raise ValueError("Requested dg_out < current dg; upsampling not supported.")

        data = self.copy(empty=False) if not inplace else self
        dg_inc = dg_out - data.dg
        if dg_inc > 0:
            data.array = self._downsample_tensor(data.array, dg_inc)
            data.dg = dg_out

        # Optionally apply a mask at the target resolution (simple multiplicative mask)
        if mask_MR is not None:
            mask = self._get_mask_at_dg(mask_MR, data.dg)
            if mask.shape[-2:] != data.array.shape[-2:]:
                raise ValueError("Mask and data have incompatible spatial shapes.")
            data.array = data.array * mask
        return data
    
    ###########################################################################
    def downsample_toMR(self, dg_max, mask_MR=None):
        """
        Generate a MR (multi-resolution) object by downsampling the current
        (single-resolution) data to all resolutions between dg=0 and dg_max.

        Only supports MR=False and assumes current dg==0.
        """
        if self.MR:
            raise ValueError("downsample_toMR expects MR == False.")
        if self.dg != 0:
            raise ValueError("downsample_toMR assumes current data is at dg=0.")
        if dg_max < 0:
            raise ValueError("dg_max must be non-negative.")
        if self.array is None:
            raise ValueError("No array stored in this object.")

        list_arrays = []
        list_dg = list(range(dg_max + 1))

        for dg in list_dg:
            if dg == 0:
                arr = self.array
            else:
                arr = self._downsample_tensor(self.array, dg)

            if mask_MR is not None:
                mask = self._get_mask_at_dg(mask_MR, dg)
                if mask.shape[-2:] != arr.shape[-2:]:
                    raise ValueError(f"Mask and data have incompatible shapes at dg={dg}.")
                arr = arr * mask

            list_arrays.append(arr)

        data = self.copy(empty=True)
        data.MR = True
        data.dg = None
        data.list_dg = list_dg
        data.array = list_arrays

        return data
    
    ###########################################################################
    def downsample_fromMR(self, Nout):
        """
        Convert an MR==True object to MR==False at resolution Nout.

        Each resolution in the current MR list is downsampled to Nout and then
        stacked into a single array of shape (..., len(list_dg), *Nout).
        """
        if not self.MR:
            raise ValueError("downsample_fromMR expects MR == True.")
        if self.array is None or len(self.array) == 0:
            raise ValueError("No data stored in this MR object.")
        if not isinstance(Nout, (tuple, list)) or len(Nout) != 2:
            raise ValueError("Nout must be a tuple (Nx_out, Ny_out).")

        Nx_out, Ny_out = Nout
        out_list = []

        for arr in self.array:
            H, W = arr.shape[-2:]
            if (H, W) == (Nx_out, Ny_out):
                y = arr
            else:
                if H % Nx_out != 0 or W % Ny_out != 0:
                    raise ValueError(f"Cannot downsample from ({H},{W}) to ({Nx_out},{Ny_out}).")
                factor_x = H // Nx_out
                factor_y = W // Ny_out
                if factor_x != factor_y:
                    raise ValueError("Anisotropic downsampling is not supported in downsample_fromMR.")
                dg_inc = int(round(math.log2(factor_x)))
                if 2 ** dg_inc != factor_x:
                    raise ValueError("Downsampling factor must be a power of 2.")
                y = self._downsample_tensor(arr, dg_inc)
            out_list.append(y)

        # stack along a new dimension before spatial dims
        stacked = torch.stack(out_list, dim=-3)

        data = self.copy(empty=True)
        data.MR = False
        data.array = stacked

        # infer dg from N0 and Nout if possible
        if self.N0 is not None:
            scale_x = self.N0[0] // Nx_out
            if scale_x > 0 and 2 ** int(round(math.log2(scale_x))) == scale_x:
                data.dg = int(round(math.log2(scale_x)))
            else:
                data.dg = None
        else:
            data.dg = None
        data.list_dg = None

        return data
    
    ###########################################################################
    def smooth(self, inplace=False):
        """Apply isotropic smoothing mirroring FoCUS.smooth 2D pathway."""

        target = self.copy(empty=False) if not inplace else self

        def _apply_smooth(tensor: torch.Tensor) -> torch.Tensor:
            *leading, Nx, Ny = tensor.shape
            ndata = int(torch.prod(torch.tensor(leading))) if leading else 1
            flat = tensor.reshape(ndata, Nx, Ny)
            smoothed = _complex_conv2d_circular(flat, self.smooth_kernel)
            return smoothed.reshape(*leading, Nx, Ny)

        if target.MR:
            target.array = [_apply_smooth(t) for t in target.array]
        else:
            target.array = _apply_smooth(target.array)

        target.dtype = target.array.dtype
        return target
    
    ###########################################################################
    def modulus(self, inplace=False):
        """
        Compute the modulus (absolute value) of the data.
        """
        data = self.copy(empty=False) if not inplace else self

        if data.MR:
            data.array = [torch.abs(a) for a in data.array]
        else:
            data.array = torch.abs(data.array)
            
        data.dtype=data.array.dtype

        return data
        
    ###########################################################################
    def mean(self, square=False, mask_MR=None):
        '''
        Compute the mean on the last two dimensions (Nx, Ny).

        If MR=True, the mean is computed for each resolution and stacked in
        an additional last dimension of size len(list_dg).

        If a multi-resolution mask is given, it is assumed to have unit mean
        at each resolution (as enforced by downsample_toMR_Mask), so the mean
        is computed as mean(x * mask).
        '''
        if self.MR:
            means = []
            for arr, dg in zip(self.array, self.list_dg):
                arr_use = torch.abs(arr) ** 2 if square else arr
                dims = (-2, -1)
                if mask_MR is not None:
                    mask = self._get_mask_at_dg(mask_MR, dg)
                    mean = (arr_use * mask).mean(dim=dims)
                else:
                    mean = arr_use.mean(dim=dims)
                means.append(mean)
            mean = torch.stack(means, dim=-1)
        else:
            if self.array is None:
                raise ValueError("No data stored in this object.")
            arr_use = torch.abs(self.array) ** 2 if square else self.array
            dims = (-2, -1)
            if mask_MR is not None:
                mask = self._get_mask_at_dg(mask_MR, self.dg)
                mean = (arr_use * mask).mean(dim=dims)
            else:
                mean = arr_use.mean(dim=dims)

        return mean
        
    ###########################################################################
    def cov(self, data2=None, mask_MR=None, remove_mean=False):
        """
        Compute the covariance between data1=self and data2 on the last two
        dimensions (Nx, Ny).

        Only works when MR == False.
        """
        if self.MR:
            raise ValueError("cov currently supports only MR == False.")

        x = self.array
        if data2 is None:
            y = x
        else:
            if not isinstance(data2, STL_2D_Kernel_Torch):
                raise TypeError("data2 must be a Planar2D_kernel_torch instance.")
            if data2.MR:
                raise ValueError("data2 must have MR == False.")
            if data2.dg != self.dg:
                raise ValueError("data2 must have the same dg as self.")
            y = data2.array

        dims = (-2, -1)

        if mask_MR is not None:
            mask = self._get_mask_at_dg(mask_MR, self.dg)
            if remove_mean:
                mx = (x * mask).mean(dim=dims, keepdim=True)
                my = (y * mask).mean(dim=dims, keepdim=True)
                x_c = x - mx
                y_c = y - my
            else:
                x_c = x
                y_c = y
            cov = (x_c * y_c.conj() * mask).mean(dim=dims)
        else:
            if remove_mean:
                mx = x.mean(dim=dims, keepdim=True)
                my = y.mean(dim=dims, keepdim=True)
                x_c = x - mx
                y_c = y - my
            else:
                x_c = x
                y_c = y
            cov = (x_c * y_c.conj()).mean(dim=dims)
            
        return cov        
       
    def get_wavelet_op(self, J=None, L=None, kernel_size=None):
        if L is None:
            L = 4
        if kernel_size is None:
            kernel_size = 5
        if J is None:
            J = np.min([int(np.log2(self.N0[0])),int(np.log2(self.N0[1]))])-3
        
        return WavelateOperator2Dkernel_torch(kernel_size,L,J,
                                              device=self.array.device,dtype=self.array.dtype)
       

class WavelateOperator2Dkernel_torch:
    def __init__(self, kernel_size: int, L: int, J: int,
                 device='cuda', dtype=torch.float):
        self.KERNELSZ = kernel_size
        self.L = L
        self.J = J
        self.device = torch.device(device)
        self.dtype = dtype
        
        self.kernel = self._wavelet_kernel(kernel_size,L)
        self.WType='simple'
        
    def _wavelet_kernel(self,kernel_size: int,n_orientation: int,sigma=1):
        """Create a 2D Wavelet kernel."""
        # coords = torch.arange(kernel_size, device=self.device, dtype=self.dtype) - (kernel_size - 1) / 2.0
        # yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        # mother_kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))[None,:,:]
        # angles=torch.arange(n_orientation, device=self.device, dtype=self.dtype)/n_orientation*torch.pi
        # angles_proj=torch.pi*(xx[None,...]*torch.cos(angles[:,None,None])+yy[None,...]*torch.sin(angles[:,None,None]))
        # kernel = torch.complex(torch.cos(angles_proj)*mother_kernel,torch.sin(angles_proj)*mother_kernel)
        # kernel = kernel - torch.mean(kernel,dim=(1,2))[:,None,None]
        # kernel = kernel / torch.sqrt(torch.sum(kernel**2, dim=(1,2)))[:,None,None]
        # return kernel.reshape(1,n_orientation,kernel_size,kernel_size)
        
        ###Morlay wavelet
        coords = torch.arange(kernel_size, device=self.device, dtype=self.dtype) - (kernel_size - 1) / 2.0
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        
        # Gaussian envelope
        gaussian_envelope = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        
        # Orientations
        angles = torch.arange(n_orientation, device=self.device, dtype=self.dtype) / n_orientation * torch.pi
        
        # Morlet wavelet: exp(i*k0*x_rot) * gaussian_envelope
        # x_rot is the coordinate along the orientation direction
        x_rot = xx[None, :, :] * torch.cos(angles[:, None, None]) + yy[None, :, :] * torch.sin(angles[:, None, None])
        
        # Complex Morlet wavelet
        kernel = torch.exp(1j * 0.75 * np.pi * x_rot ) * gaussian_envelope[None, :, :]
        
        # Remove DC component (admissibility condition)
        kernel = kernel - torch.mean(kernel, dim=(1, 2))[:, None, None]
        
        # L2 normalization
        kernel = kernel / torch.sqrt(torch.sum(torch.abs(kernel)**2, dim=(1, 2)))[:, None, None]
        
        return kernel.reshape(1, n_orientation, kernel_size, kernel_size)
            
    def get_L(self):
        return self.L
        
    def apply(self, data,j):
        """
        Apply the convolution kernel to data.array [..., Nx, Ny]
        and return cdata [..., L, Nx, Ny].

        Parameters
        ----------
        data : object
            Object with an attribute `array` storing the data as a tensor
            or numpy array with shape [..., Nx, Ny].

        Returns
        -------
        torch.Tensor
            Convolved data with shape [..., L, Nx, Ny].
        """
        if j!=data.dg :
            raise 'j is not equal to dg, convolution not possible'
            
        x = data.array  # [..., Nx, Ny]

        # Ensure x is a torch tensor on the same device / dtype as the kernel
        x = torch.as_tensor(x, device=self.kernel.device, dtype=self.kernel.dtype)

        weight = self.kernel.squeeze(0)  # [L, K, K]

        convolved = _complex_conv2d_circular(x, weight)

        return STL_2D_Kernel_Torch(convolved,smooth_kernel=data.smooth_kernel,dg=data.dg,N0=data.N0)
        
    
    def apply_smooth(self, data: STL_2D_Kernel_Torch, inplace: bool = False):
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
        x = data.array  # [..., K]=
        *leading, K1,K2 = x.shape

        # Flatten leading dims into batch dimension: (B, Ci=1, K)
        if leading:
            B = int(np.prod(leading))
        else:
            B = 1
        x_bc = x.reshape(B, 1, K1,K2)

        # Smooth kernel (Ci=1, Co=1, P)
        w_smooth = self.kernel.abs().to(device=data.device, dtype=data.dtype)

        y_bc = _conv2d_circular(x, w_smooth)
        
        if not isinstance(y_bc, torch.Tensor):
            y_bc = torch.as_tensor(y_bc, device=data.device, dtype=data.dtype)

        y = y_bc.reshape(*leading, K1,K2)  # same shape as input x

        # Copy or in-place update
        out = data.copy(empty=True) if not inplace else data
        out.array = y
        # metadata stays identical (nside, N0, dg, cell_ids, ...)
        return out
