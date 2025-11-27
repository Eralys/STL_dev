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
    
    #smooth kernel coded
    # x,y=np.meshgrid(np.arange(5)-2,np.arange(5)-2)
    # sigma=1.0
    # np.exp(-(x**2+y**2)/(2*sigma**2))
    smooth_kernel = np.array([[0.01831564, 0.082085  , 0.13533528, 0.082085  , 0.01831564],
                            [0.082085  , 0.36787944, 0.60653066, 0.36787944, 0.082085  ],
                            [0.13533528, 0.60653066, 1.        , 0.60653066, 0.13533528],
                            [0.082085  , 0.36787944, 0.60653066, 0.36787944, 0.082085  ],
                            [0.01831564, 0.082085  , 0.13533528, 0.082085  , 0.01831564]])
                                     
    ###########################################################################
    def __init__(self, array):
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
        
            
    ###########################################################################
    @staticmethod
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
        elif isinstance(array, np.ndarray):
            return torch.from_numpy(array).to('cuda')
        elif isinstance(array, torch.Tensor):
            return array.to('cuda')
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
    def downsample(self, dg_out, mask_MR=None, copy=False):
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
    def modulus(self, copy=False):
        """
        Compute the modulus (absolute value) of the data.
        """
        data = self.copy(empty=False) if copy else self

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
       
    def get_wavelet_op(self, kernel_size=None,L=None,J=None):
        
        if L is None:
            L=4
        if kernel_size is None:
            kernel_size=5
        if J is None:
            J=np.min([int(np.log2(self.N0[0])),int(np.log2(self.N0[1]))])-3
        
        return WavelateOperator2Dkernel_torch(kernel_size,L,J,
            device=self.array.device,dtype=self.array.dtype)
       

class WavelateOperator2Dkernel_torch:
    def __init__(self, kernel_size: int, L: int, J: int, device='cuda',dtype=torch.float):
        """
        kernel: torch.Tensor
            Convolution kernel, either of shape [1, L, K, K] .
            L is the number of output channels.
        """
        self.device=device
        self.dtype=dtype
        
        self.kernel = self._wavelet_kernel(kernel_size,L)
        self.L=L
        self.J=J
        self.WType='simple'
        
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

        cdata = STL_2D_Kernel_Torch(cdata)
        cdata.dg=data.dg
        cdata.N0=data.N0
        return cdata
        
