"""
Created on Wed Nov 14:07 2018
"""
import numpy as np
import torch
from copy import deepcopy

class STL_2D_FFT_Torch:
    """
    Class for 2D planar STL FFT using PyTorch
    """
    
    @staticmethod
    def copy_array(data):
        """
        Copies input PyTorch array.
        """
        array = copy(data)
        return array

    @staticmethod    
    def cov(array1, fourier_status1, array2, fourier_status2, mask = None, remove_mean=False):
        """
        Compute the covariance of two tensors on their last two dimensions.
        
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
        array1 : torch.Tensor (complex or real)
            First array whose covariance has to be computed.
        fourier_status1 : Bool
            Fourier status of array1
        array2 : torch.Tensor (complex or real)
            Second array whose covariance has to be computed.
        fourier_status2 : Bool
            Fourier status of array2
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
    
        if remove_mean:
            raise Exception("Remove mean is yet not implemented.")
        
        if mask is None and fourier_status1 and fourier_status2:
            # Compute covariance (complex values)
            cov =  torch.mean(array1 * array2.conj(), dim=(-2, -1))
        else:
            # We pass everything to real space
            if fourier_status1:
                _array1 = torch.fft.ifft2(array1, norm="ortho")
            else:
                _array1 = array1
            if fourier_status2:
                _array2 = torch.fft.ifft2(array2, norm="ortho")
            else:
                _array2 = array2
            # Define mask
            mask = 1 if mask is None else mask
            # Compute covariance (complex values)
            cov =  torch.mean(_array1 * _array2.conj() * mask, dim=(-2, -1))
                
        return cov
    

    def __init__(self, data, fourier_status=False):
        """
        Initialize the STL_2D_FFT_torch class.

        fourier_status: True if data is in Fourier space.
        """
        self.array = self.__class__.to_array(data) if data is not None else None
        self.fourier_status = fourier_status

        self.DT = '2D_FFT_Torch'
        self.MR = False
        self.dg = 0
        self.N0 = self.findN()


    def findN(self):
        """
        Find the dimensions of the 2D planar data, which are expected to be the 
        last two dimensions of the array.

        Returns
        -------
        N : tuple of int
            The spatial dimensions  of the 2D planar data.
        """
        return tuple(self.array.shape[-2:])
        

    def copy(self, empty=False):
        """
        Copy an instance.
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
        
        return self.__class__(self.array if not empty else None, 
                              fourier_status=self.fourier_status)
    

    
    def modulus(self, inplace=False):
        """
        Take the modulus of the array attribute.

        Parameters
        ----------
        inplace: bool
            if inplace is True, overwrites self.array by its modulus.
        
        Returns
        -------
        torch.Tensor
            Modulus of input tensor.
        """
        
        if inplace:
            output = self
        else:
            output = self.copy()

        output.array = output.array.abs()
    
        return output

    def mean(self, square=False, mask=None):
        """
        Compute the mean of the tensor on its last two dimensions.
        
        A mask in real space can be given. It should be of unit mean.
        
        Parameters
        ----------
        array : torch.Tensor
            Input tensor whose mean has to be computed.
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
        
        # Define unit mask if no mask is given
        mask = 1 if mask is None else mask 

        if self.fourier_status is True:
            raise Exception("Mean in Fourier space is yet not implemented.") 
        else: # Real space
            if square==False:
                return torch.mean(self.array * mask, dim=(-2, -1))
            else:
                return torch.mean((self.array.abs())**2 * mask, dim=(-2, -1)) 
    
   
    def fourier(self):
        """
        Compute the Fourier Transform on the last two dimensions of the input 
        tensor.

        Parameters
        ----------
        array : torch.Tensor
            Input tensor for which the Fourier Transform is to be computed.

        Returns
        -------
        torch.Tensor
            Fourier transform of the input tensor along the last two dimensions.
        """
        if self.fourier_status:
            return self.array
        else:
            return torch.fft.fft2(self.array, norm="ortho")
    
    def ifourier(self):
        """
        Compute the inverse Fourier Transform on the last two dimensions of the input 
        tensor.

        Parameters
        ----------
        array : torch.Tensor
            Input tensor for which the inverse Fourier Transform is to be computed.

        Returns
        -------
        torch.Tensor
            Inverse Fourier transform of the input tensor along the last two dimensions.
        """
        if not self.fourier_status:
            return self.array
        else:
            return torch.fft.ifft2(self.array, norm="ortho")

    def subsampling(self, dg_out, O_fourier = False, mask_MR=None):
        """
        Downsample the self.array to the specified resolution.
        
        Note: Masks are not supported in this data type.
        
        Parameters
        ----------
        dg_out : int
            Desired downsampling factor of the data.
        mask_MR : None
            Placeholder for mask, not used in this function.
        
        Returns
        -------
        torch.Tensor
            Downsampled data at the desired downgrading factor dg_out.
        fourier_status : bool
            Indicates whether output array is in Fourier space.        
        """
        
        if mask_MR is not None:
            raise Exception("Masks are not supported in DT1") 
            
        if dg_out == self.dg:
            return self.array, self.fourier_status
        
        # Tuning parameter to keep the aspect ratio and a unified resolution
        min_x, min_y = 8, 8
        if self.N0[0] > self.N0[1]:
            min_x = int(min_x * self.N0[0]/self.N0[1])
        elif self.N0[1] > self.N0[0]:
            min_y = int(min_y * self.N0[1]/self.N0[0])

        # Identify the new dimensions
        dx = int(max(min_x, self.N0[0] // 2**(dg_out + 1)))
        dy = int(max(min_y, self.N0[1] // 2**(dg_out + 1)))
        
        # Check expected current dimensions
        dx_cur = int(max(min_x, self.N0[0] // 2**(self.dg + 1)))
        dy_cur = int(max(min_y, self.N0[1] // 2**(self.dg + 1)))
        
        # Perform downsampling if necessary
        if dx != dx_cur or dy != dy_cur:
            
            # Fourier transform if in real space
            if not self.fourier_status:
                array = self.fourier()
                self.fourier_status = True
            else:
                array = self.array
                        
            # Downsampling in Fourier
            array_dg = torch.cat(
                (torch.cat(
                    (array[...,:dx, :dy], array[...,-dx:, :dy]), -2),
                torch.cat(
                    (array[...,:dx, -dy:], array[...,-dx:, -dy:]), -2)
                ),-1) * np.sqrt(dx * dy / dx_cur / dy_cur)
            
            self.array = array_dg
            return self.out_fourier(O_fourier)

        else:
            return self.array
        
    def out_fourier(self, O_fourier):
        """
        Put the  in the desired Fourier status (O_Fourier).
        
        Parameters
        ----------
        - O_fourier : bool
            Desired Fourier status (True = Fourier space, False = real space)
        - copy : bool
            If True, returns a new stl_array instance.
        """
        
        # Check that O_Fourier is a bool
        if not isinstance(O_fourier, bool):
            raise Exception("0_Fourier should be a bool")
    
        # If current status differs from desired
        if self.fourier_status != O_fourier:
            if O_fourier:
                data = self.fourier()
            else:
                data = self.ifourier()
        # Already in desired space
        else:
            data = self.copy()
    
        return data
    
        

    def get_wavelet_op(self, J=None, L=4, WType="Crappy"):
    
        # Default values
        if J is None:
            J = int(np.log2(min(self.N0))) - 2

        # Wtype-specific construction
        if WType == "Crappy":
            return CrappyWavelateOperator2D_FFT_torch(J, L, self.N0)
        else:
            raise Exception("Wavelet type not yet supported.")

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
    
class CrappyWavelateOperator2D_FFT_torch:
    '''
    Class whose instances correspond to a wavelet transform operator.
    The wavelet set and the operator is built during the initilization.
    The operator is applied through apply method.
    This method is DT-dependent, and actually calls independent iterations,
    but with common method and attribute structure.
    
    The multi-resolution is dealt with several parameters:
        dg_max, which indicates the maximum dg resolution
        j_to_dg, which indicate the dg_j resolution associated to each j scale.
        
    For example, if you work with J=6 wavelets and N0=256, you can have:
        - dg_max=4, with associated dg factors (0, 1, 2, 3, 4)
        - true downsampling factors (1, 2, 4, 8, 16)
        - actual resolutions (256, 128, 64, 32, 16)
        - a j_to_dg list (0, 0, 1, 1, 2, 3) associated to j in range(J=6)
    
    The wavelet convolution is DT-dependent, an is performed either in real or
    Fourier space.
    
    They are two main types of wavelet arrays.
        - Single_Kernel==True: In this case, a set of L oriented wavelets is 
        defined at a single pixellation. Convolutions at different scales are
        then down by subsequent susampling and convolution in pixel space with
        this single set of L oriented wavelets, and convolutions at all scales
        can not be done at the initial N0 resolution.
        - SR_Kernel==False: In this case, a set of J*L dilated and rotated
        wavelets is defined at the initial N0 resolution, and a convolution at 
        all scales at this initial resolution can be performed. Convolution in
        a multi-resolution scheme can also be done, where convolution at each
        scale is done at the proper downsampling factor.
        
    This mean that different wavelets need to be stored:
        - Single_Kernel==True: a single set of L wavelets, stored in the  
        wavelet_array attribute.
        - SR_Kernel==False: a set of J*L wavelets, store both at the 
        initial N0 resolution in wavelet_array, and in a multi-resolution 
        framework in wavelet_array_MR.
    => The wavelet_j method then allows to call for the correct quantity when
        a convolution is performed.
        
    Rq: when Single_Kernel==True, j_to_dg = range(J). This is not necessarily 
        the case when False.

    See apply for more details.

    Parameters
    ----------
    - DT : str
        Type of data (1d, 2d planar, HealPix, 3d)
    - N0 : tuple
        initial size of array (can be multiple dimensions)
    - J : int
        number of scales
    - L : int
        number of orientations
    - WType : str
        type of wavelets (e.g., "Morlet" or "Bump-Steerable")
    
    Attributes 
    ----------
    - parent parameters (DT,N0,J,L,WType)
    - dg_max: int  (DT- and WType-dependent)
        maximum dg resolution of the Wavelet Transform
        (DT- and WType-dependent)
    - j_to_dg : list of int  
        list of actual dg_j resolutions at each j scale 
    - wavelet_array : torch tensor 
        * array of wavelets at L orientation if Single_Kernel==True.
        * array of wavelets at J*L scales and orientation at N0 resolution
        if Single_Kernel==False
    - wavelet_array_MR : list (len J) of arrays
        list of arrays of L wavelets at all J scales and at Nj resolution
        Only if if Single_Kernel==False.
    - Single_Kernel : bool
        if convolution done at all scales with the same L oriented wavelets
    - mask_opt : bool
        If it is possible to do use masked during the convolution

    Questions and to do
    ----------
    - Is Single_Kernel sufficient in itself? We could separate the fact to do
    the convolution with a kernel in real space and to be able to do all
    convolution at the initial N0 resolution, using two different attribute.
    - We could similarly attach the fact to use mask to the fact to have
    Single_Kernel==True.
    - Do we add a low-pass filter per default for j=J ? 
    - Do we impose j_to_dg = range(J) for simplicity and efficiency?
    - I propose to anyway only have dyadic wavelets for the "main set".
        -> the inclusion of P00' power spectrum terms can be done differently.
    - for __init__, we could ask either for DT and N, or for a stl_array
      instance, from which DT and N are obtained. Could be added if useful.
    - A proper "by the book" set of Wavelets should be implemented, with proper
    Littelwood-Paley and co conditions.
      
    '''

    @staticmethod
    def gaussian_bank(J, L, size, base_mu=None, base_sigma=None):
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


    def __init__(self, J, L, N0):
        '''
        Constructor, see details above.
        '''
        # Main parameters
        self.DT = '2D_FFT_Torch'
        self.N0 = N0
        self.J = J
        self.L = L
        self.WType = "Crappy"
        self.wavelet_array = None
        self.wavelet_array_MR = None
        self.dg_max = None
        self.j_to_dg = None
        self.Single_Kernel = None
        self.mask_opt = None
    
        # Build all the wavelets-related attributes.
        # Also fix J, L, and WType values if None.
        self.build()

        
    def build(self):
        '''
        Build wavelet set and subsampling_factors, see details above.
        The standard values for J, L, and WType are also fixed if None.
        '''
        
        # Build the wavelet set

        # Create the full resolution Wavelet set
        self.wavelet_array = self.__class__.gaussian_bank(self.J, self.L, self.N0)
        
        # Find dg_max (with a min size of 16 = 2 * 8)
        # To avoid storing tensors at the same effective resolution
        self.dg_max = int(np.log2(min(self.N0)) - 4)
        
        # Create the MR list of wavelets
        self.wavelet_array_MR = []
        self.j_to_dg = []
        for j in range(self.J):
            dg = min(j, self.dg_max)
            subsampled_wavelet_array, fourier_status = STL_2D_FFT_Torch(data=self.wavelet_array[j],
                                                                        fourier_status=True).subsampling(dg_out=dg)
            assert fourier_status
            self.wavelet_array_MR.append(subsampled_wavelet_array)
            self.j_to_dg.append(dg)

            
    ###########################################################################
    def wavelet_j(self, j):
        """
        Return the necessary wavelets to perform the convolution at scale j.
        
        If Single_Kernel==True, always return the same set of L wavelets.
        If Single_Kernel==False, return the L wavelets at scale j and at the
        Nj resolution.
        """
    
        if self.Single_Kernel:
            return self.wavelet_array
        else:
            return self.wavelet_array_MR[j]
            
    ###########################################################################
    def plot(self, Fourier=None):
        '''
        Plot the set of wavelets, either in Fourier or real space.
        Can add a selection of (j,l).
        '''
    
        # To be done

    @staticmethod
    def wavelet_conv_full(data, wavelet_set, fourier_status, mask=None):
        """
        Perform a convolution of data by the entire wavelet set at full resolution.
        
        No mask is allowed in this DT.

        Parameters
        ----------
        - data : torch.Tensor of size (..., N0)
            Data whose convolution is computed
        - wavelet_set : torch.Tensor of size (J, L, N0)
            Wavelet set
        - fourier_status:
            Fourier status of the data
        - mask : torch.Tensor of size (...,N0) -> None expected
            Mask for the convolution

        Returns
        -------
        - conv: torch.Tensor (..., J, L, N0)
            Convolution between data and wavelet_set
        - fourier_status: bool 
            Fourier status of the convolution (True in this DT)
        """
        
        if mask is not None:
            raise NotImplementedError("Mask is not yet allowed in this DT")

        # Pass data in Fourier if in real space
        _data = data if fourier_status else torch.fft.fft2(data)
        
        # Compute the convolution
        conv = _data[..., None, None, :, :] * wavelet_set
        
        # Fourier status related to the DT
        fourier_status = True
        
        return conv, fourier_status
    
    @staticmethod
    def wavelet_conv(data, wavelet_j, fourier_status, mask_MR=None):
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
        - fourier_status:
            Fourier status of the data
        - mask_MR : list of torch.Tensor of size (...,Nj) -> None expected
            Multi-resolution masks for the convolution

        Returns
        -------
        - conv: torch.Tensor (..., L, N0)
            Convolution between data and wavelet_set at scale j
        - fourier_status: bool 
            Fourier status of the convolution (True in this DT)
        """
        if mask_MR is not None:
            raise NotImplementedError("Mask is not yet allowed in this DT")
        
        # Pass data in Fourier if in real space
        _data = data if fourier_status else torch.fft.fft2(data)
        
        # Compute the convolution
        conv = _data[..., None, :, :] * wavelet_j
        
        # Fourier status related to the DT
        fourier_status = True
        
        return conv, fourier_status
        
    ###########################################################################
    def apply(self, data, MR=None, j=None, mask_MR=None, O_Fourier=None):
                  
        '''
        Compute the Wavelet Transform (WT) of data.
        This method is DT dependent, and calls independent iterations with 
        common method and attribute structure.
        
        Data should be a MR==False StlData instance. The wavelet transform can
        either be computed at all J*L scales and angles (fullJ), or at a given
        j scale (single_j) for all L orientations?
        
        For code efficiency, this method requires a MR=True StlData instance 
        for the masks at all resolution, with list_dg = range(dg_max + 1).
        
        The different modes are:
        - fullJ (j=None and MR=False): convolution at J*L scales and angles, 
            without a MR framework, only if Single_Kernel==False.
            Input data should be a MR=False StlData instance at 
            dg=0 resolution. No mask are a priori allowed in this case.
        - fullJ_MR (j=None and MR=False): convolution at J*L scales and angles,
            within a MR framework, always possible.
            Input data should be a MR=True StlData instance at all
            resolution between dg=0 and dg_max [ lids_dg = range(dg_max) ]
        - single_j: at L angles at a given scale j, within a MR framework.
            Input data should be a MR=False StlData instance wtih dg = dg_j.
        
        Rq: If j = None, the defaut value for MR if j=None is False if 
        Single_Kernel==False, and True else. For a single_j convolution, MR
        can only be true.
        Rq: mask_MR is allowed only if mask_opt==True.
           
        Parameters
        ----------
        - data : StlData, 
            Input data of same DT/N0, can be batched on several dimension.
            -> MR=False dg=0 if fullJ 
            -> MR=True list_dg=range(dg_max+1) if fullJ_MR 
            -> MR=False dg=dg_j if single_j 
        - MR : bool
            If convolution to be done in a MR framework. 
        - j : int
            Scale at which the convolution is done. Done at all scales if None.
        - mask_MR : StlData with MR=True or None
            Multi-resolution masks, requires list_dg = range(dg_max + 1)
            mask is not allowed in fullJ mode
        - O_Fourier : bool or None
            Desired Fourier status of output. 
            If None, DT-dependent default is used.
            
        Output 
        ----------
        - WT : StlData:
            -> MR==False (..., J, L, N0) if j is None and MR==False
            -> MR==True list of (..., L, Nj) if j is None and MR==False
            -> (..., L, Nj) if j == int 
            Wavelet convolutions at different scales and angles. 
            
        Questions and to do
        ----------
        - I propose not to deal with the issue of non-periodicity here, but 
        only in the mean and cov functions, at the end of the computations.
        - We could think at the possibility to compute WT at fixed (j,l)
        values, if it helps distributing the computations for large batchs.
        - To decide if we impose a condition on mask_MR, like the fact that it
        is on unit mean.
        - I'm a bit skeptical by the fact that an internal Fourier transform
        could be necessary here, since it means that the same transform could
        have to be on multiple call of this method.
        - For the convolution at a fixed scale. Should we accept data that are 
        not at Nj resolution and downsample them? It need to be see with usage
        '''
        
        # Check coherence of input data.
        if not isinstance(data, STL_2D_FFT_Torch):
            raise Exception("Data should be a STL_2D_FFT_Torch instance") 
        if self.DT != data.DT:
            raise Exception("Data and wavelet transform should have same DT")
        if self.N0 != data.N0:
            raise Exception("Data and wavelet transform should have same N0")
            
        # Check coherence of mask.
        if mask_MR is not None:
            if not self.mask_opt:
              raise Exception(
                  "Wavelet transform with masks not supported for this DT") 
            if not isinstance(mask_MR, STL_2D_FFT_Torch):
                raise Exception("Mask should be a STL_2D_FFT_Torch instance") 
            if self.DT != mask_MR.DT:
                raise Exception(
                    "Mask and wavelet transform should have same DT")
            if self.N0 != mask_MR.N0:
                raise Exception(
                    "Mask and wavelet transform should have same N0")    
            if mask_MR.list_dg != list(range(self.dg_max+1)):
                raise Exception(
                    "Mask should be between MR between dg=0 and dg_max \n" \
                    "Use downsample_toMR_Mask method") 
            if mask_MR.fourier_status:
                raise Exception("Mask should be in real space") 
        
        # Set MR default-value if None:
        if MR is None:
            MR = True if self.Single_Kernel else False
            
        # Convolution at all scales at the same time
        if j is None:

            # fullJ
            if MR==False:
                # Check valid Single_Kernel value
                if self.Single_Kernel:
                    raise Exception("Convolutions at all scales with MR==False"
                                    "not supported with this DT") 
                # Check that resolutions are compatible
                if data.dg != 0:
                    raise Exception("Data should be at dg=0 resolution") 
                # Create a new output STL_2D_FFT_Torch instance 
                # and compute the WT, all DT are not necessarily included here.
                WT = STL_2D_FFT_Torch(*self.__class__.wavelet_conv_full(
                    data.array, self.wavelet_array, data.fourier_status, 
                    None if mask_MR is None else mask_MR.array[0]))
            
            # fullJ_MR
            elif MR==True:
                raise Exception("Not implemented yet")
                # # Check that resolutions are compatible

                # if data.list_dg != list(range(self.dg_max+1)):
                #     raise Exception(
                #         "Data should be between MR between dg=0 and dg_max \n"
                #         "Use downsample_toMR method") 
                # # Create the ouptut stl.array instance for the WT
                # WT = StlData._init_MR(self.DT, None, self.N0, self.j_to_dg)
                # # Compute the WT, all DT are not necessarily included here.
                # wavelet_conv_full_MR = {
                #                 "DT1": DT1_wavelet_conv_full_MR, 
                #                 "DT2": DT2_wavelet_conv_full_MR
                #                 }.get(self.DT)
                # WT.array, WT.Fourier = wavelet_conv_full_MR(
                #              data.array, self.wavelet_array_MR,
                #              data.Fourier, self.j_to_dg,
                #              None if mask_MR is None else mask_MR.array)
            
        # Convolution at a given j scale in MR (single_j)
        else:
            if not isinstance(j, int):
                raise Exception("j should be a single int") 
            # Check that dg_j resolutions are compatible
            if data.dg != self.j_to_dg[j]:
                raise Exception("Data should be at dg_j resolution") 
            # Create the autput stl_array instance for the Wavelet Transform
            # and compute the WT at a given j
            WT = STL_2D_FFT_Torch(*self.__class__.wavelet_conv(
                data.array, self.wavelet_j(j), data.fourier_status, 
                None if mask_MR is None else mask_MR.array[j]))
       
        # Transform to correct Fourier status if necessary
        if O_Fourier is not None:
            WT.out_fourier(O_Fourier)

        return WT
