# -*- coding: utf-8 -*-
"""
Main structure of STL

Tentative proposal by EA 
"""

from StlData import StlData
from DataType1 import DT1_wavelet_build, DT1_wavelet_conv, \
                      DT1_wavelet_conv_full, DT1_wavelet_conv_full_MR
from DataType2 import DT2_wavelet_build, DT2_wavelet_conv,\
                      DT2_wavelet_conv_full_MR
         
###############################################################################
###############################################################################
    
class Wavelet_Operator:
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

    ###########################################################################
    def __init__(self, DT, N0, J=None, L=None, WType=None):
        '''
        Constructor, see details above.
        '''
        # Main parameters
        self.DT = DT
        self.N0 = N0
        self.J = J
        self.L = L
        self.WType = WType
        self.wavelet_array = None
        self.wavelet_array_MR = None
        self.dg_max = None
        self.j_to_dg = None
        self.Single_Kernel = None
        self.mask_opt = None
    
        # Build all the wavelets-related attributes.
        # Also fix J, L, and WType values if None.
        self.build()
        
    ###########################################################################
    def build(self):
        '''
        Build wavelet set and subsampling_factors, see details above.
        The standard values for J, L, and WType are also fixed if None.
        '''
        
        # Built the wavelet set
        wavelet_build = {"DT1": DT1_wavelet_build, \
                     "DT2": DT2_wavelet_build}.get(self.DT)
        (self.wavelet_array, self.wavelet_array_MR, self.dg_max, self.j_to_dg,
         self.Single_Kernel, self.mask_opt,
         self.J, self.L, self.WType) = wavelet_build(
            self.N0, self.J, self.L, self.WType)
            
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
        if not isinstance(data, StlData):
            raise Exception("Data should be a StlData instance") 
        if self.DT != data.DT:
            raise Exception("Data and wavelet transform should have same DT")
        if self.N0 != data.N0:
            raise Exception("Data and wavelet transform should have same N0")
            
        # Check coherence of mask.
        if mask_MR is not None:
            if not self.mask_opt:
              raise Exception(
                  "Wavelet transform with masks not supported for this DT") 
            if not isinstance(mask_MR, StlData):
                raise Exception("Mask should be a StlData instance") 
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
            if mask_MR.Fourier:
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
                # Create a new autput stl.StlData instance 
                WT = data.copy(empty=True)
                # Compute the WT, all DT are not necessarily included here.
                wavelet_conv_full = {"DT1": DT1_wavelet_conv_full}.get(self.DT)
                WT.array, WT.Fourier = wavelet_conv_full(
                            data.array, self.wavelet_array, data.Fourier, 
                            None if mask_MR is None else mask_MR.array[0])
            
            # fullJ_MR
            elif MR==True:
                # Check that resolutions are compatible

                if data.list_dg != list(range(self.dg_max+1)):
                    raise Exception(
                        "Data should be between MR between dg=0 and dg_max \n"
                        "Use downsample_toMR method") 
                # Create the ouptut stl.array instance for the WT
                WT = StlData._init_MR(self.DT, None, self.N0, self.j_to_dg)
                # Compute the WT, all DT are not necessarily included here.
                wavelet_conv_full_MR = {
                                "DT1": DT1_wavelet_conv_full_MR, 
                                "DT2": DT2_wavelet_conv_full_MR
                                }.get(self.DT)
                WT.array, WT.Fourier = wavelet_conv_full_MR(
                             data.array, self.wavelet_array_MR,
                             data.Fourier, self.j_to_dg,
                             None if mask_MR is None else mask_MR.array)
            
        # Convolution at a given j scale in MR (single_j)
        else:
            if not isinstance(j, int):
                raise Exception("j should be a single int") 
            # Check that dg_j resolutions are compatible
            if data.dg != self.j_to_dg[j]:
                raise Exception("Data should be at dg_j resolution") 
            # Create the autput stl_array instance for the Wavelet Transform
            WT = data.copy(empty=True)
            # Compute the WT at a given j
            wavelet_conv = {"DT1": DT1_wavelet_conv, \
                     "DT2": DT2_wavelet_conv}.get(self.DT)
            WT.array, WT.Fourier = wavelet_conv(
                data.array, self.wavelet_j(j), data.Fourier, 
                None if mask_MR is None else mask_MR.array[j])
       
        # Transform to correct Fourier status if necessary
        if O_Fourier is not None:
            WT.out_fourier(O_Fourier)

        return WT