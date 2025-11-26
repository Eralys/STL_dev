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

import numpy as np
import torch
import torch.nn.functional as F

###############################################################################
###############################################################################
class STL2DKernel:
    '''
    Class which contain the different types of data used in STL.
    Store important parameters, such as DT, N0, and the Fourier type.
    Also allow to convert from numpy to pytorch (or other type).
    Allow to transfer internally these parameters.
    
    Has different standard functions as methods (fourier_func, fourier_func, 
    out_fourier, modulus, mean_func, cov_func, downsample)
    
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
        - fourier_func, fourier_func, out_fourier and modulus_func are applied 
          to each array of the list.
        - mean_func, cov_func give a single vector or last dim len(list_N)
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
    - Fourier : bool
         if data is in Fourier
         
    '''
    
    ###########################################################################
    def __init__(self, array, smooth_kernel=None):
        '''
        Constructor, see details above. Frontend version, which assume the 
        array is at N0 resolution with dg=0, with MR=False.
        
        More sophisticated Back-end constructors (_init_SR and _init_MR) exist.
        
        '''
        
        # Check that MR==False array is given
        if isinstance(array, list):
            raise ValueError("Only single resolution array are accepted.")
        
        # Main 
        
        self.MR = False
        self.dg = 0
        self.Nx = array.shape[-2]
        self.Ny = array.shape[-1]
        
        self.list_dg = None
        
        self.array = self.to_array(array)
        
        # Find N0 value
        
        self.device='cuda'
        self.dtype=torch.float
        if smooth_kernel is None:
            smooth_kernel=self._smooth_kernel(3)
        self.smooth_kernel=smooth_kernel
        
        
    def _smooth_kernel(self,kernel_size: int):
        """Create a 2D Gaussian kernel."""
        sigma=1
        # 1D coordinate grid centered at 0
        coords = torch.arange(kernel_size, device=self.device, dtype=self.dtype) - (kernel_size - 1) / 2.0
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        return kernel
            
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
            return array
        elif isinstance(array, np.ndarray):
            return torch.from_numpy(array)
        elif isinstance(array, torch.Tensor):
            return array
        else:
            raise ValueError("Input must be a NumPy array or PyTorch tensor.")
            
    ###########################################################################
    def copy(self, empty=False):
        """
        Copy a Planar2D_kernel_torch instance.
        Array is put to None if empty==True.
        
        Parameters
        ----------
        - empty : bool
            If True, set array to None.
                    
        Output 
        ----------
        - Planar2D_kernel_torch
           copy of self
        """
        new = object.__new__(Planar2D_kernel_torch)

        # Copy metadata
        new.MR = self.MR
        new.N0 = self.N0
        new.dg = self.dg
        new.list_dg = list(self.list_dg) if self.list_dg is not None else None
        new.Fourier = self.Fourier
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
        dimensions using average pooling.

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

        orig_shape = x.shape
        x_flat = x.reshape(-1, 1, H, W)
        y = F.avg_pool2d(x_flat, kernel_size=(scale, scale), stride=(scale, scale))
        H2, W2 = H // scale, W // scale
        y = y.reshape(*orig_shape[:-2], H2, W2)
        return y
  
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
    def downsample(self, dg_out, mask_MR=None, O_Fourier=None, copy=False):
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

        data = self.copy(empty=False) if copy else self
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
    def downsample_toMR(self, dg_max, mask_MR=None, O_Fourier=None):
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
    def downsample_fromMR(self, Nout, O_Fourier=None):
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
    def modulus_func(self, copy=False):
        """
        Compute the modulus (absolute value) of the data.
        """
        data = self.copy(empty=False) if copy else self

        if data.MR:
            data.array = [torch.abs(a) for a in data.array]
        else:
            data.array = torch.abs(data.array)

        return data
        
    ###########################################################################
    def mean_func(self, square=False, mask_MR=None):
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
    def cov_func(self, data2=None, mask_MR=None, remove_mean=False):
        """
        Compute the covariance between data1=self and data2 on the last two
        dimensions (Nx, Ny).

        Only works when MR == False.
        """
        if self.MR:
            raise ValueError("cov_func currently supports only MR == False.")

        x = self.array
        if data2 is None:
            y = x
        else:
            if not isinstance(data2, Planar2D_kernel_torch):
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
            cov = (x_c * y_c * mask).mean(dim=dims)
        else:
            if remove_mean:
                mx = x.mean(dim=dims, keepdim=True)
                my = y.mean(dim=dims, keepdim=True)
                x_c = x - mx
                y_c = y - my
            else:
                x_c = x
                y_c = y
            cov = (x_c * y_c).mean(dim=dims)
            
        return cov        
       
    def get_wavelet_op(self,kernel_size=5,L=4):
        
        return WavelateOperator2Dkernel_torch(kernel_size,L,device=self.array.device,dtype=self.array.dtype)
       

class WavelateOperator2Dkernel_torch:
    def __init__(self, kernel_size: int, L=4, device='cuda',dtype=torch.float):
        """
        kernel: torch.Tensor
            Convolution kernel, either of shape [1, L, K, K] .
            L is the number of output channels.
        """
        self.device=device
        self.dtype=dtype
        
        self.kernel = self._wavelet_kernel(kernel_size,L)
        
    def _wavelet_kernel(self,kernel_size: int,n_orientation: int,sigma=1):
        """Create a 2D Wavelet kernel."""
        # 1D coordinate grid centered at 0
        coords = torch.arange(kernel_size, device=self.device, dtype=self.dtype) - (kernel_size - 1) / 2.0
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        mother_kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))[None,:,:]
        angles=torch.arange(n_orientation, device=self.device, dtype=self.dtype)/n_orientation*np.pi
        angles_proj=torch.pi*(xx[None,...]*torch.cos(angles[:,None,None])+yy[None,...]*torch.sin(angles[:,None,None]))
        kernel = torch.complex(torch.cos(angles_proj)*mother_kernel,torch.sin(angles_proj)*mother_kernel)
        kernel = kernel - torch.mean(kernel,dim=(1,2))[:,None,None]
        kernel = kernel / torch.sum(kernel.abs(),dim=(1,2))[:,None,None]
        return kernel.reshape(1,n_orientation,kernel_size,kernel_size)
            

    def apply(self, data):
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
        x = data.array  # [..., Nx, Ny]

        # Ensure x is a torch tensor on the same device / dtype as the kernel
        x = torch.as_tensor(x, device=self.kernel.device, dtype=self.kernel.dtype)

        # Separate leading dimensions and spatial dimensions
        *leading_dims, Nx, Ny = x.shape

        # Flatten all leading dims into a single batch dimension for conv2d
        # After this, x_4d has shape [B, 1, Nx, Ny], with B = prod(leading_dims)
        if leading_dims:
            B = 1
            for d in leading_dims:
                B *= d
        else:
            B = 1

        x_4d = x.reshape(B, 1, Nx, Ny)

        # Prepare the kernel for torch.nn.functional.conv2d
        k = self.kernel
        if k.dim() != 4:
            raise ValueError(f"Kernel must have 4 dimensions, got shape {k.shape}.")

        c0, c1, Kh, Kw = k.shape

        # We accept kernels of shape [1, L, K, K] or [L, 1, K, K]
        # and convert them to [L, 1, K, K] for conv2d.
        if c0 == 1:
            # Kernel is [1, L, K, K] -> permute to [L, 1, K, K]
            weight = k.permute(1, 0, 2, 3).contiguous()
            out_channels = c1
        elif c1 == 1:
            # Kernel is already [L, 1, K, K]
            weight = k
            out_channels = c0
        else:
            raise ValueError(
                f"Kernel shape {k.shape} is not compatible with "
                "expected [1, L, K, K] or [L, 1, K, K]."
            )

        # Use 'same' padding manually: assuming K is odd
        pad_h = Kh // 2
        pad_w = Kw // 2

        # Perform the 2D convolution:
        # x_4d:    [B, 1, Nx, Ny]
        # weight:  [L, 1, Kh, Kw]
        # result:  [B, L, Nx, Ny] (because of padding)
        y_4d = F.conv2d(x_4d, weight, padding=(pad_h, pad_w))

        # Reshape back to original leading dims plus channel L and spatial dims
        # y_4d has shape [B, L, Nx, Ny]
        new_shape = (*leading_dims, out_channels, Nx, Ny)
        cdata = y_4d.reshape(new_shape)

        return STL2DKernel(cdata,smooth_kernel=data.smooth_kernel)
        

'''  
###############################################################################
def to_array(array):
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
        return array
    elif isinstance(array, np.ndarray):
        return torch.from_numpy(array)
    elif isinstance(array, torch.Tensor):
        return array
    else:
        raise ValueError("Input must be a NumPy array or PyTorch tensor.")

###############################################################################
def findN(array, Fourier):
    """
    Find the dimensions of the 2D planar data, which are expected to be the 
    last two dimensions of the array.

    Parameters
    ----------
    array : torch.Tensor
        Input tensor whose spatial dimensions need to be determined.
    Fourier : bool
        Indicates whether the array is in Fourier space.
        Not used here.

    Returns
    -------
    N : tuple of int
        The spatial dimensions  of the 2D planar data.
    """
    
    # Get the shape of the tensor
    shape = array.shape
    # Return the last two dimensions
    return (shape[-2], shape[-1])
    
###############################################################################

def copy(array, N0, dg):
    """
    Copy a PyTorch tensor.

    Parameters
    ----------
    array : torch.Tensor
        Input tensor to be copied.
    
    Returns
    -------
    torch.Tensor
        A copy of the input tensor.
    """
    
    return array.clone()

###############################################################################

def modulus(array):
    """
    Take the modulus (absolute value) of a tensor.

    Parameters
    ----------
    array : torch.Tensor
        Input tensor.
    
    Returns
    -------
    torch.Tensor
        Modulus of input tensor.
    """
    
    return array.abs()

###############################################################################

def DT1_fourier(array, N0, dg):  
    """
    ??????
    Compute the Fourier Transform on the last two dimensions of the input 
    tensor.

    Parameters
    ----------
    array : torch.Tensor
        Input tensor for which the Fourier Transform is to be computed.
    N0 : tuple of int
        Initial resolution of the data, not used.
    dg : int
        Current downsampling factor of the data, not used.

    Returns
    -------
    torch.Tensor
        Fourier transform of the input tensor along the last two dimensions.
    """
    
    return torch.fft.fft2(array, norm="ortho")

###############################################################################

def DT1_ifourier(array, N0, dg):
    """
    ????
    Compute the inverse Fourier Transform on the last two dimensions of the 
    input tensor and return the real part of the result.
    
    Parameters
    ----------
    array : torch.Tensor
        Input tensor for which the inverse Fourier Transform is to be computed.
    N0 : tuple of int
        Initial resolution of the data, not used.
    dg : int
        Current downsampling factor of the data, not used.

    Returns
    -------
    torch.Tensor
        Real part of the inverse Fourier transform of the input tensor along
        the last two dimensions.
    """
    
    return torch.fft.ifft2(array, norm="ortho").real

###############################################################################
def Mask_toMR(mask, N0, dg_max):
    """
    Return an error, since masks are not supported in this data type.
    """
    
    return mask_MR
    
###############################################################################
def subsampling_func(import torch
import torch.nn.functional as F
from typing import Tuple, Optional


def downsample_torch(
    array: torch.Tensor,
    N0: Tuple[int, int],
    dg: int,
    dg_out: int,
    mask_MR: Optional[torch.Tensor] = None,
):
    """
    Downsample the data to the specified resolution.

    Note: Masks are not supported in this data type.

    Parameters
    ----------
    array : torch.Tensor
        Input tensor to be downsampled. The last two dimensions are assumed to be
        the spatial dimensions.
    N0 : tuple of int
        Initial resolution of the data (height, width).
    dg : int
        Current downsampling factor of the data (relative to N0).
    dg_out : int
        Desired downsampling factor of the data (relative to N0).
    mask_MR : None
        Placeholder for mask, not used in this function.

    Returns
    -------
    torch.Tensor
        Downsampled data at the desired downgrading factor dg_out.
    bool
        Indicates whether output array is in Fourier space (always False here).
    """
    if mask_MR is not None:
        raise ValueError("Masks are not yet supported in this function (mask_MR must be None).")

    # No change requested
    if dg_out == dg:
        return array, False

    # Only support further downsampling (coarser resolution)
    if dg_out < dg:
        raise ValueError(
            f"dg_out ({dg_out}) must be >= dg ({dg}) for downsampling. "
            "Upsampling is not supported here."
        )

    if dg_out % dg != 0:
        raise ValueError(
            f"dg_out ({dg_out}) must be an integer multiple of dg ({dg})."
        )

    factor = dg_out // dg

    # Basic consistency check with the initial resolution
    H0, W0 = N0
    if H0 % dg_out != 0 or W0 % dg_out != 0:
        raise ValueError(
            f"N0={N0} is not compatible with dg_out={dg_out} "
            "(N0 must be divisible by dg_out)."
        )

    if array.ndim < 2:
        raise ValueError("Input tensor must have at least 2 dimensions (spatial).")

    *leading, H, W = array.shape

    if H % factor != 0 or W % factor != 0:
        raise ValueError(
            f"Current spatial size ({H}, {W}) is not divisible by the downsampling "
            f"factor {factor}."
        )

    # Reshape to 4D for avg_pool2d: (N, C, H, W)
    reshaped = array.reshape(-1, 1, H, W)

    if torch.is_complex(reshaped):
        real = F.avg_pool2d(reshaped.real, kernel_size=factor, stride=factor)
        imag = F.avg_pool2d(reshaped.imag, kernel_size=factor, stride=factor)
        down = torch.complex(real, imag)
    else:
        down = F.avg_pool2d(reshaped, kernel_size=factor, stride=factor)

    H_out, W_out = H // factor, W // factor
    down = down.reshape(*leading, H_out, W_out)

    fourier = False  # this routine operates in direct space
    return down, fourier
    
###############################################################################
def DT1_subsampling_func_toMR(array, Fourier, N0, dg_max, mask_MR):
    """
    Generate a list of downsampled input array from resolution dg=0 to 
    dg=dg_max, following list_dg = range(dg_max + 1). Input array is expected 
    at dg=0 resolution.
    
    Note: Masks are not supported in this data type.
    
    Parameters
    ----------
    array : torch.Tensor
        Input tensor to be downsampled.
    Fourier : bool
        Indicates whether the array is in Fourier space.
    N0 : tuple of int
        Initial resolution of the data.-
    dg_max : int
        Maximum downsampling factor
    mask_MR : None
        Placeholder for mask, not used in this function.
    
    Returns
    -------
    list of torch.Tensor
        List of downsampled tensors for each downgrading factor from dg=0 to 
        dg=dg_max.
    fourier : bool
        Indicates whether output array is in Fourier space.     
    """
    
    if mask_MR is not None:
        raise Exception("Masks are not supported in DT1")

    # First Fourier transform if necessary.
    if not Fourier:
        array = torch.fft.fft2(array, norm="ortho")
        Fourier = True
        
    downsampled_arrays = [array]
    current_array = array

    for dg_out in range(1, dg_max + 1):
        current_array, _ = DT1_subsampling_func(
            current_array, Fourier, N0, dg_out - 1, dg_out, None)
        downsampled_arrays.append(current_array)

    return downsampled_arrays, Fourier

###############################################################################
def DT1_mean_func(array, N0, dg, square, mask):
    """
    Compute the mean of the tensor on its last two dimensions.
    
    A mask in real space can be given. It should be of unit mean.
    
    Parameters
    ----------
    array : torch.Tensor
        Input tensor whose mean has to be computed.
    N0 : tuple of int
        Initial resolution of the data (not used in this function).
    dg : int
        Current downsampling factor of the data (not used in this function).
    square : bool
        If True, compute the quadratic mean.
    mask : torch.Tensor, optional
        Mask tensor whose last dimensions should match with input array.
        It should be of unit mean.

    Returns
    -------
    torch.Tensor
        Mean of input array on the last two dimensions.
    """
    
    # Define mask
    mask = 1 if mask is None else mask

    # Compute mean
    if square is False:
        return torch.mean(array * mask, dim=(-2, -1))
    else:
        return torch.mean((array.abs())**2 * mask, dim=(-2, -1)) 

###############################################################################
def DT1_mean_func_MR(array, N0, list_dg, square, mask_MR):
 
    """
    Compute the mean of a list of tensors on their last two dimensions.
    The other dimensions of the tensors must match.
    
    These means are stacked on the last dimension of the output tensor.
    
    A multi-resolution mask in real space can be given. It should be of unit 
    mean at each resolution.
 
    Parameters
    ----------
    array : list of torch.Tensor
        List of input tensors for which the mean is to be computed.
    N0 : tuple of int
        Initial resolution of the data (not used in this function).
    list_dg : list of int
        List of downsampling factors of the data (not used in this function).
    square : bool
        If True, compute the quadratic mean.
    mask_mr : list of torch.Tensor, optional
        List of mask tensors at the relevant resolutions.
        Last dimensions should match with input array.
        They should be of unit mean at each resolution.
 
    Returns
    -------
    torch.Tensor
        Mean of input arrays, stacked on the last dimension.
    """
     
    # Pre-allocate the resulting tensor
    shape_except_N = array[0].shape[:-2]
    len_list = len(array)
    mean = torch.empty(shape_except_N + (len_list,))
    
    # Loop the mean computation over the list
    for i, tensor in enumerate(array):
        # Define mask
        mask = 1 if mask_MR is None else mask_MR[i]
        
        # Compute mean
        if square is False:
            mean[..., i] = torch.mean(
                array[i] * mask, dim=(-2, -1)) 
        else:
            mean[..., i] =  torch.mean(
                (array[i].abs())**2 * mask, dim=(-2, -1))

    return mean

###############################################################################
def DT1_cov_func(array1, Fourier1, array2, Fourier2, 
                 N0, dg, mask, remove_mean):
    """
    Compute the covariance of two tensors on their last two dimension.
    
    Covariance can be computed either in real space of in Fourier space.
    if mask is None:
        - in real space if they are both in real space
        - in Fourier space if they are both in Fourier space
        - in real space if they are in different space
    else:
        - in real space
        
    A mask in real space can be given. It should be of unit mean.
    
    The mean of array1 and array2 are removed before the covariance computation
    only if remove_mean = True.

    Parameters
    ----------
    array1 : torch.Tensor
        First array whose covariance has to be computed.
    Fourier1 : Bool
        Fourier status of array1
    array2 : torch.Tensor
        Second array whose covariance has to be computed.
    Fourier1 : Bool
        Fourier status of array2
    N0 : tuple of int
        Initial resolution of the data (not used in this function).
    dg : int
        Current downsampling factor of the data (not used in this function).
    mask : torch.Tensor, optional
        Mask tensor whose last dimensions should match with input array.
        It should be of unit mean.
    
    Returns
    -------
    torch.Tensor
        Cov of input array1 and array2 on the last two dimensions.
        
    Remark and to do
    -------
    - Remove_mean = True not implemented. To be seen if this is necessary.
    """
        
    if mask is None and Fourier1 and Fourier2:
        # Compute covariance (complex values)
        cov =  torch.mean(array1 * array2.conj(), dim=(-2, -1)).real
    else:
        # We pass everything to real space
        if Fourier1:
            _array1 = torch.fft.ifft2(array1, norm="ortho").real
        else:
            _array1 = array1
        if Fourier2:
            _array2 = torch.fft.ifft2(array2, norm="ortho").real
        else:
            _array2 = array2
        # Define mask
        mask = 1 if mask is None else mask
        # Compute covariance (complex values)
        cov =  torch.mean(_array1 * _array2 * mask, dim=(-2, -1))
            
    return cov

###############################################################################
def DT1_wavelet_build(N0, J, L, WType):
    """
    Generate a set of 2D planar wavelets in Fourier space, both in full 
    resolution and in a multi-resolution settings, as well as the related
    parameters.
    
    Default values for J, L, and Wtype are used if None.
    
    Parameters
    ----------
    - N0 : tuple
        initial size of array (can be multiple dimensions)
    - J : int
        number of scales
    - L : int
        number of orientations
    - WType : str
        type of wavelets (e.g., "Morlet" or "Bump-Steerable")
    
    Returns
    -------
    wavelet_array : torch.Tensor of size (J,L,N0)
        Array of wavelets at J*L scales and orientation at N0 resolution.
    wavelet_array_MR : list of torch.Tensor of size (L,Nj)
        list of arrays of L wavelets at all J scales and at Nj resolution.
    dg_max : int
        Maximum dg downsampling factor
    - j_to_dg : list of int  
        list of actual dg_j resolutions at each j scale 
    - Single_Kernel : bool -> False here
        If convolution done at all scales with the same L oriented wavelets
    - mask_opt : bool -> False here
        If it is possible to do use masked during the convolution

    """
    
    # Default values
    if J is None:
        J = int(np.log2(min(N0))) - 2
    if L is None:
        L = 4
    if WType is None:
        WType = "Crappy"
    
    # Wtype-specific construction
    if WType == "Crappy":
        # Crappy wavelet set for test. A proper one should be implemented.
        
        # Create the full resolution Wavelet set
        wavelet_array = gaussian_bank(J, L, N0)
        
        # Find dg_max (with a min size of 16 = 2 * 8)
        # To avoid storing tensors at the same effective resolution
        dg_max = int(np.log2(min(N0)) -4)
        
        # Create the MR list of wavelets
        wavelet_array_MR = []
        j_to_dg = []
        for j in range(J):
            dg = min(j, dg_max)
            wavelet_array_MR.append(DT1_subsampling_func(
                wavelet_array[j], True, N0, 0, dg, None)[0])
            j_to_dg.append(dg)

    # Values of Single_Kernel and mask_opt
    Single_Kernel = False
    mask_opt = False
    
    return (wavelet_array, wavelet_array_MR, 
            dg_max, j_to_dg, Single_Kernel, mask_opt,
            J, L, WType)

###############################################################################
def gaussian_2d_rotated(mu, sigma, angle, size):
    """
    Generate a rotated 2D Gaussian centered at an offset mu along the rotated
    axis from image center.

    Parameters
    ----------
    mu : float
        Offset along the rotated axis from the image center (in pixels).
    sigma : float
        Isotropic standard deviation (spread).
    angle : float
        Rotation angle in radians (0 to pi).
    size : tuple of int
        Grid size (M, N) = (height, width).

    Returns
    -------
    torch.Tensor
        A 2D Gaussian (M, N) with unit L2 norm.
    """
    
    M, N = size
    x = torch.linspace(0, M - 1, M)
    y = torch.linspace(0, N - 1, N)
    X, Y = torch.meshgrid(x, y, indexing='ij')

    # Image center
    cx = M / 2
    cy = N / 2

    # Compute offset from center along rotated axis
    cos_a = torch.cos(torch.tensor(angle))
    sin_a = torch.sin(torch.tensor(angle))
    center_x = cx - mu * sin_a
    center_y = cy + mu * cos_a

    # Gaussian centered at (center_x, center_y)
    G = torch.exp(-((X - center_x)**2 + (Y - center_y)**2) / (2 * sigma**2))
    
    # Threshold
    eps = 10**-1
    G[G < eps] = 0
    
    return G

###############################################################################
def gaussian_bank(J, L, size, base_mu = None, base_sigma = None):
    """
    Generate a bank of rotated and scaled 2D Gaussians.

    Parameters
    ----------
    J : int
        Number of dyadic scales.
    L : int
        Number of orientations.
    base_sigma : float
        Smallest sigma (spread).
    base_mu : float
        Base offset along the rotated axis.
    size : tuple of int
        Grid size (M, N).

    Returns
    -------
    torch.Tensor
        A tensor of shape (J, L, M, N), each entry L2-normalized.
    """
    M, N = size
    filters_bank = torch.empty((J, L, M, N))

    if base_mu is None:
        base_mu = min(M, N) / (2*torch.sqrt(torch.tensor(2.0)))
    if base_sigma is None:
        base_sigma = base_mu / (2*torch.sqrt(torch.tensor(2.0)))

    for j in range(J):
        sigma = base_sigma / (2 ** j)
        mu = base_mu / (2 ** j)
        for l in range(L):
            angle = float(l) * torch.pi / L
            filters_bank[j, l] = gaussian_2d_rotated(mu, sigma, angle, size)
            
    # Return the zero frequency to (0,0), and put it to zero
    filters_bank = torch.fft.fftshift(filters_bank, dim=(-2, -1))
    filters_bank[:,:,0,0] = 0

    return filters_bank

###############################################################################
def DT1_wavelet_conv_full(data, wavelet_set, Fourier, mask):
    """
    Perform a convolution of data by the entire wavelet set at full resolution.
    
    No mask is allowed in this DT.

    Parameters
    ----------
    - data : torch.Tensor of size (..., N0)
        Data whose convolution is computed
    - wavelet_set : torch.Tensor of size (J, L, N0)
        Wavelet set
    - Fourier:
        Fourier status of the data
    - mask : torch.Tensor of size (...,N0) -> None expected
        Mask for the convolution

    Returns
    -------
    - conv: torch.Tensor (..., J, L, N0)
        Convolution between data and wavelet_set
    - Fourier: bool 
        Fourier status of the convolution (True in this DT)
    """
    
    # Pass data in Fourier if in real space
    _data = data if Fourier else torch.fft.fft2(data)
    
    # Compute the convolution
    conv = _data[..., None, None, :, :] * wavelet_set
    
    # Fourier status related to the DT
    Fourier = True
    
    return conv, Fourier

###############################################################################
def DT1_wavelet_conv_full_MR(data, wavelet_set, Fourier, j_to_dg, mask_MR):   
    """
    Perform a convolution of data by the entire wavelet in a multi-resolution
    setting. 
    
    A multi-resolution mask can be given.

    Parameters
    ----------
    - data : list of torch.Tensor of size (..., Nj)
        Multi-resolution data whose convolution is computed.
        The associated dg are list_dg = range(dg_max + 1)
    - wavelet_set : list of torch.Tensor of size (J, L, Nj)
        Multi-resolution wavelet set.
        The associated dg are j_to_dg
    - Fourier:
        Fourier status of the data
    - j_to_dg : list of int  
        list of actual dg_j resolutions at each j scale 
     - mask_MR : list of torch.Tensor of size (...,Nj) -> None expected
        Multi-resolution masks for the convolution

    Returns
    -------
    - conv: list of torch.Tensor (..., L, Nj)
        Convolution between data and wavelet_set
    - Fourier: bool 
        Fourier status of the convolution (True in this DT)
    """
    
    # Initialize conv
    conv = []
    
    for j in range(len(wavelet_set)):
        # Pass data in Fourier if in real space
        dg = j_to_dg[j]
        _data_j = data[dg] if Fourier else torch.fft.fft2(data[dg])
        
        # Compute the convolution
        conv.append(_data_j[..., None, :, :] * wavelet_set[j])
    
    # Fourier status related to the DT
    Fourier = True
    
    return conv, Fourier

###############################################################################
def DT1_wavelet_conv(data, wavelet_j, Fourier, mask_MR):
    """
    Perform a convolution of data by the wavelet at a given scale and L 
    orientation. Both the data and the wavelet should be at the Nj resolution.
    
    No mask is allowed in this DT.

    Parameters
    ----------
    - data : torch.Tensor of size (..., Nj)
        Data whose convolution is computed, at resolution Nj
    - wavelet_set : torch.Tensor of size (L, Nj)
        Wavelet set at scale j
    - Fourier:
        Fourier status of the data
     - mask_MR : list of torch.Tensor of size (...,Nj) -> None expected
        Multi-resolution masks for the convolution

    Returns
    -------
    - conv: torch.Tensor (..., L, N0)
        Convolution between data and wavelet_set at scale j
    - Fourier: bool 
        Fourier status of the convolution (True in this DT)
    """
    
    # Pass data in Fourier if in real space
    _data = data if Fourier else torch.fft.fft2(data)
    
    # Compute the convolution
    conv = _data[..., None, :, :] * wavelet_j
    
    # Fourier status related to the DT
    Fourier = True
    
    return conv, Fourier

###############################################################################
###############################################################################
###############################################################################
###############################################################################
    
def DT1_subsampling_func_fromMR(param):   
    pass
'''
