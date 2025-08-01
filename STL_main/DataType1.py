#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 27 14:23:20 2025

Example methods for a test data type.

2D planar maps with convolution in Fourier.

This class makes all computations in torch.

Characteristics:
    - in pytorch
    - assume real maps (in real space)
    - N0 gives x and y sizes for array shaped (..., Nx, Ny).
    - masks are not supported in convolutions
"""

import numpy as np
import torch

###############################################################################
def DT1_to_array(array):
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
def DT1_findN(array, Fourier):
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

def DT1_copy(array, N0, dg):
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

def DT1_modulus(array):
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
def DT1_Mask_toMR(mask, N0, dg_max):
    """
    Return an error, since masks are not supported in this data type.
    """
    
    raise Exception("Masks are not supported in DT1") 
    
###############################################################################
def DT1_subsampling_func(array, Fourier, N0, dg, dg_out, mask_MR):
    """
    Downsample the data to the specified resolution.
    
    Note: Masks are not supported in this data type.
    
    Parameters
    ----------
    array : torch.Tensor
        Input tensor to be downsampled.
    Fourier : bool
        Indicates whether input array is in Fourier space.
    N0 : tuple of int
        Initial resolution of the data.
    dg : int
        Current downsampling factor of the data.
    dg_out : int
        Desired downsampling factor of the data.
    mask_MR : None
        Placeholder for mask, not used in this function.
    
    Returns
    -------
    torch.Tensor
        Downsampled data at the desired downgrading factor dg_out.
    fourier : bool
        Indicates whether output array is in Fourier space.        
    """
    
    if mask_MR is not None:
        raise Exception("Masks are not supported in DT1") 
        
    if dg_out == dg:
        return array, Fourier
    
    # Tuning parameter to keep the aspect ratio and a unified resolution
    min_x, min_y = 8, 8
    if N0[0] > N0[1]:
        min_x = int(min_x * N0[0]/N0[1])
    elif N0[1] > N0[0]:
        min_y = int(min_y * N0[1]/N0[0])

    # Identify the new dimensions
    dx = int(max(min_x, N0[0] // 2**(dg_out + 1)))
    dy = int(max(min_y, N0[1] // 2**(dg_out + 1)))
    
    # Check expected current dimensions
    dx_cur = int(max(min_x, N0[0] // 2**(dg + 1)))
    dy_cur = int(max(min_y, N0[1] // 2**(dg + 1)))
    
    # Perform downsampling if necessary
    if dx != dx_cur or dy != dy_cur:
        
        # Fourier transform if in real space
        if not Fourier:
            array = torch.fft.fft2(array, norm="ortho")
            Fourier = True
        
        # Downsampling in Fourier
        array_dg = torch.cat(
            (torch.cat(
                (array[...,:dx, :dy], array[...,-dx:, :dy]), -2),
              torch.cat(
                (array[...,:dx, -dy:], array[...,-dx:, -dy:]), -2)
            ),-1) * np.sqrt(dx * dy / dx_cur / dy_cur)
        return array_dg, Fourier
        
    else:
        return array, Fourier
    
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