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

###############################################################################
###############################################################################
class 2DPlanar_kernel_torch:
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
    def __init__(self, array, N0=None, Fourier=False):
        '''
        Constructor, see details above. Frontend version, which assume the 
        array is at N0 resolution with dg=0, with MR=False.
        
        More sophisticated Back-end constructors (_init_SR and _init_MR) exist.
        
        '''
        
        # Check that MR==False array is given
        if isinstance(array, list):
            raise ValueError("Only single resolution array are accepted.")
        
        # Main 
        self.DT = DT
        self.MR = False
        self.dg = 0
        self.N0 = N0
        self.list_dg = None
        self.Fourier = Fourier
        
        # Put array in the correct library (torch, tensorflow...)
        to_array = {"DT1": DT1_to_array, "DT2": DT2_to_array}.get(self.DT)
        self.array = to_array(array)
        
        # Find N0 value
        findN = {"DT1": DT1_findN, "DT2": DT2_findN}.get(self.DT)
        if N0==None:
            self.N0 = findN(self.array, self.Fourier)

        
    ###########################################################################
    def copy(self, empty=False):
        """
        Copy a stl_array instance.
        Array is put to None if empty==True.
        
        Parameters
        ----------
        - empty : bool
            If True, set array to None.
                    
        Output 
        ----------
        - StlData
           copy of self
        """
        
        return data

    ###########################################################################
    def __getitem__(self, key):
        """
        To slice directly the array attribute. Produce a view of array, to 
        match with usual practices, allowing to conveniently pass only part
        of a StlData instance.
        
        Additional copy method should be applied if necessary:
            data = data[3,:,:].copy()
        
        When MR==False, we slice the multi-dimensional array:
            data2 = data1[:, :, 3, :]
        When MR==False, we slice the list of arrays:
            data2 = data1[:3] (one single dimension)
        
        To modify directly self.array, one can also simply do
        data.array = data.array[:,:,:3]
        
        Parameters
        ----------
        - key : slicing
            slicing of the array attribute
            Only one slice if self.array is a list
            
        Remark
        ----------
        - When slicing a MR=False element, there is no clear way of dealing
        with N0 and dg, if the slicing is done on the dimensions related to N. 
        -> Maybe not allow this option? Or try to protect it?
            
        """
        
        
        return data

  
    ###########################################################################
    def downsample_toMR_Mask(self, dg_max):
        '''
        Take a mask given at a dg=0 resolution, and put it at all resolutions
        from dg=0 to dg=dg_max, in a MR=True StlData.
        
        The input map should only contains real positive values, describing the
        relative weight of each pixel.
        
        Parameters
        ----------
        - self : StlData with MR=False
            Mask, should have dg=0 and Fourier=False.
            Can be batched
        - dg_max : maximum downsampling
            
        Return
        ----------
        - mask_MR : StlData with MR=True 
            Multi-resolution masks, with list_dg = range(dg_max + 1)
            Is of unit mean at each dg resolution.
            Can be batched
            
        To do and remark
        ----------
        - Should we impose that the output mask at each dg resolution should be
        of unit mean? While this is important for mean and cov, and could be 
        imposed when preparing the mask for the scattering operator, it has to 
        be seen for the wavelet convolutions. Anyway, if such a condition is 
        necessary, it should be imposed here for code efficiency.
        '''
        
            
        return Mask_MR

    ###########################################################################
    def downsample(self, dg_out, mask_MR=None, O_Fourier=None, copy=False):
        """
        Downsample the data to the dg resolution.
        Only supporte MR == False.
        
        A multi-resolution mask can be given, wih resolutions between dg=0 and 
        at least dg_out.
    
        The output Fourier status can be specified via O_Fourier.
        If not specified, it will be chosen for minimal computation cost
        (depending on DT).
    
        Parameters
        ----------
        - dg_out : int
            Target dg resolution.
        - mask_MR : StlData with MR=True or None
            Multi-resolution masks, requires list_dg = range(dg_max + 1)
            Can be batched if dimensions match
        - O_Fourier : bool or None
            Desired Fourier status of output. 
            If None, DT-dependent default is used.
        - copy : bool
            If True, return a new StlData instance; else modify in place.
    
        Returns
        -------
        data : StlData with MR=False
            Downsampled data at dg=dg_out
        """
    
    
        return data
    
    ###########################################################################
    def downsample_toMR(self, dg_max, mask_MR=None,
                        O_Fourier=None):
        """
        Generate a MR (multi-resolution) StlData object by downsampling the 
        current (single-resolution) data to a list of target resolutions.
        
        Downsample the data to all resolutions between dg=0 to dg_max.
        Only supporte MR=False, and the output is a MR=True array.
        
        A multi-resolution mask can be given, wih resolutions between dg=0 and 
        at least dg_max.
    
        The output Fourier status can be specified via O_Fourier.
        If not specified, it will be chosen for minimal computation cost
        (depending on DT).
    
        Parameters
        ----------
        - dg_max : int
            Maximum dg resolution to reach
        - mask_MR : StlData with MR=True or None
            Multi-resolution masks, requires list_dg = range(dg_max + 1)
            Can be batched if dimensions match
        - O_Fourier : bool or None
            Desired Fourier status of output. 
            If None, DT-dependent default is used.
            
        Returns
        -------
        data : StlData with MR=True
            Downsampled data between dg=0 and dg_max
        """
    
        return data
    
    ###########################################################################
    def downsample_fromMR(self, Nout, O_Fourier=None):
        """
        Not up to date.
        Will be updated if necessary.
        
        Convert an MR==True StlData object to MR==False at at Nout.
    
        Each resolution in the current MR list is downsampled to Nout and then
        stacked into a single array of shape (..., len(listN), *Nout).
    
        Parameters
        ----------
        - Nout : tuple
            Target resolution for the final single-resolution data.
        - O_Fourier : bool or None
            Desired Fourier status of output. 
            If None, DT-dependent default is used.
    
        Returns
        -------
        data : StlData
            A new MR == False StlData object with stacked downsampled data.
    
        """
    
        
        return data
    
    ###########################################################################
    def modulus_func(self, copy=False):
        """
        Compute the modulus (absolute value) of the data.
        Automatically transforms to real space if needed.
        
        Parameters
        ----------
        copy : bool
            If True, returns a new StlData instance.
        """
        
                    
        return data
        
    ###########################################################################
    def mean_func(self, square=False, mask_MR=None):
        '''
        Compute the mean of an StlData instance on the tuple N last dimensions.
        Mean is computed in real-space, and iFourier is applied if necessary.
        
        If MR=True, the mean will be computed on the data at each resolution, 
        and put in a additional dimension of size len(list_dg) at the end.
        -> this requires that all element of the StlData objects have same
        dimension but the N ones.
        
        A multi-resolution mask can be given, wih resolutions between dg=0 and 
        at least dg_max. 
        -> At each resolution, the mask should be of unit mean, to allow for a
        proper weighting of the mean.
        
        A quadratic mean |x|^2 is computed if square = True.

        Parameters
        ----------
        - square : bool
            True if quadratic mean
        - mask_MR : StlData with MR=True or None
            Multi-resolution masks, requires list_dg = range(dg_max + 1).
            Should be of unit mean at each dg resolution.
            Can be batched if dimensions match
            
        Output 
        ----------
        - mean : array (...)  
            mean of data on last dim N
            
        Remark 
        ----------
        - The computation of the mean in real space could be done directly in 
        Fourier space if necessary (k=0 value), if there is no mask. But I am
        not sure that this use actually appears.
        - The fact that the mask is of unit mean is required, in order not to 
        compute again this mean at each call of the function.
        '''    
        
                    
        return mean
        
    ###########################################################################
    def cov_func(self, data2=None, mask_MR=None, remove_mean=False):
        """
        Compute the covariance between data1=self and data2 on the tuple N 
        last dimension.
        
        Notes:
        - Only works when MR == False. Raises an error otherwise.
        - Resolutions dg of data1 and data2 must match.
        - Automatically handles Fourier vs real space, depending on DT.
        
        A multi-resolution mask can be given, wih resolutions between dg=0 and 
        at least dg_max. 
        -> At each resolution, the mask should be of unit mean, to allow for a
        proper weighting of the mean.
        
        -> Depending on the data type (DT), the covariance can be computed in 
        real space, Fourier space, or both (e.g., using Plancherel's theorem 
        for 2D planar data). The function applies the appropriate transform 
        if needed. If a mask is provided, the computation is always performed 
        in real space.
                
        Parameters
        ----------
        - data2 : StlData with MR=False or None
            Second data. Auto-covariance of self is computed if None.
        - mask_MR : StlData with MR=True or None
            Multi-resolution masks, requires list_dg = range(dg_max + 1).
            Should be of unit mean at each dg resolution.
            Can be batched if dimensions match
        - remove_mean : bool
            If mean should be explicitely removed. 
    
        Returns
        -------
        - cov : array (...)
            Covariance value.
            
        Remark and to do
        -------
        - This function estimate the covariance without removing the mean of 
        each component. This is sufficient when at least one of the component
        is of zero mean, which is usually the case when computing ST
        statistics, and save a lot of computations.
        -> I added an option if mean should be explicitly removed, if this 
        appears to be relevant at some point.
        -> Technically, it could be better to remove the mean when we work
        with masked data, of with non-pbc. However, I think that not computing
        it could still be a good compromise.
        - The fact that the mask is of unit mean is required, in order not to 
        compute again this mean at each call of the function.
            
        """
        
        return cov
        
        
        def get_wavelet_op(self,J=None,L=None):
            return wop_class
            
sigma=1

def _smooth_kernel(kernel_size: int):
    """Create a 2D Gaussian kernel."""
    # 1D coordinate grid centered at 0
    coords = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2.0
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel
    
def _wavelet_kernel(kernel_size: int,n_orientation: int):
    """Create a 2D Gaussian kernel."""
    # 1D coordinate grid centered at 0
    coords = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2.0
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel
    
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