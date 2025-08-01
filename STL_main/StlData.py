# -*- coding: utf-8 -*-
"""
Main structure of STL

Tentative proposal by EA 
"""

from DataType1 import DT1_cov_func,\
                      DT1_subsampling_func, DT1_fourier, DT1_ifourier,\
                      DT1_mean_func, DT1_findN, DT1_modulus,\
                      DT1_subsampling_func_toMR, DT1_subsampling_func_fromMR,\
                      DT1_Mask_toMR, DT1_to_array, DT1_copy, DT1_mean_func_MR
from DataType2 import DT2_cov_func,\
                      DT2_subsampling_func, DT2_fourier, DT2_ifourier,\
                      DT2_mean_func, DT2_findN, DT2_Mask_toMR,\
                      DT2_modulus,\
                      DT2_subsampling_func_toMR, DT2_subsampling_func_fromMR,\
                      DT2_to_array, DT2_copy, DT2_mean_func_MR
         
###############################################################################
###############################################################################
class StlData:
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
    def __init__(self, DT, array, N0=None, Fourier=False):
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
    @classmethod
    def _init_SR(cls, DT, array, N0, dg=0, Fourier=False):
        '''
        Internal constructor for MR=False StlData objects.
        
        '''
        
        # Construct the SR StlData object
        data = cls(DT, array, N0=N0, Fourier=Fourier) 
        data.dg = dg
        
        return data
        
    ###########################################################################
    @classmethod
    def _init_MR(cls, DT, array_list, N0, list_dg, Fourier=False):
        '''
        Internal constructor for MR=True StlData objects.
        
        '''
        
        # Construct the MR StlData object
        data = cls(DT, None, N0=N0, Fourier=Fourier) 
        data.MR = True
        data.dg = None
        data.list_dg = list_dg
        data.array = array_list
        
        # Put array in the correct library (torch, tensorflow...)
        to_array = {"DT1": DT1_to_array, "DT2": DT2_to_array}.get(DT)
        if array_list is not None:
            data.array = list(map(to_array, array_list))
        
        return data
        
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
        
        # New array value, which can be a copy
        copy = {"DT1": DT1_copy, "DT2": DT2_copy}.get(self.DT)
        array = None if empty else copy(self.array)
        
        # MR-dependent initialization.
        if not self.MR:
            data=StlData._init_SR(
                self.DT, array, self.N0, self.dg, self.Fourier)
        else:
            data=StlData._init_MR(
                self.DT, array, self.N0, self.list_dg, self.Fourier)
        
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
        
        # Create the stl array
        data = self.copy(empty=True)
        
        if self.MR==True:
            # Slice the list
            data.array = self.array[key]
            data.list_dg = self.list_dg[key]
            
            # Transform in a MR=False array if necessary
            if isinstance(data.list_dg, int):
                data.MR = False
                data.array = data.array
                data.dg = data.list_dg
                data.list_dg = None
                
        else:
            # Convert to tuple if it is a single item
            if not isinstance(key, tuple):
                key = (key,)
            # Slice the array
            data.array = self.array[key]
            
        return data

    ###########################################################################
    def fourier_func(self, copy=False):
        '''
        Compute the Fourier transform of the data.
        Generalizes to harmonic transforms for spherical data.
        
        Parameters
        ----------
        - copy : bool
            If True, return a new StlData instance; else modify in place.
        '''    

        if self.Fourier:
            raise Exception("Data already in Fourier space.")
    
        # Create a new instance if copy=True
        data = self.copy(empty=True) if copy else self
    
        # Get Fourier transform function
        fourier = {"DT1": DT1_fourier, "DT2": DT2_fourier}.get(self.DT)
    
        if self.MR:
            # Apply Fourier transform to each resolution level
            data.array = [
            fourier(a, self.N0, dg) for a, dg in zip(self.array, self.list_dg)
            ]
        else:
            # Single-resolution case
            data.array = fourier(self.array, self.N0, self.dg)
    
        # Update Fourier status
        data.Fourier = True
    
        return data
            
    ###########################################################################
    def ifourier_func(self, copy=False):
        """
        Compute the inverse Fourier transform of data.
        Generalizes to harmonic transforms for spherical data.
    
        Parameters
        ----------
        - copy : bool
            If True, return a new StlData instance; else modify in place.
        """
    
        if self.Fourier == False:
            raise Exception("Data already in real space.")
    
        # Create new instance if copy=True
        data = self.copy(empty=True) if copy else self
    
        # Get inverse Fourier transform function
        ifourier = {"DT1": DT1_ifourier, "DT2": DT2_ifourier}.get(self.DT)
    
        if self.MR:
            # Apply inverse Fourier to each resolution level
            data.array = [
            ifourier(a, self.N0, dg) for a, dg in zip(self.array, self.list_dg)
            ]
        else:
            # Single-resolution case
            data.array = ifourier(self.array, self.N0, self.dg)
    
        # Update Fourier status
        data.Fourier = False
    
        return data
            
    ###########################################################################
    def out_fourier(self, O_Fourier, copy=False):
        """
        Put the StlData in the desired Fourier status (O_Fourier).
        If already in the correct space, optionally return a copy.
        
        Parameters
        ----------
        - O_Fourier : bool
            Desired Fourier status (True = Fourier space, False = real space)
        - copy : bool
            If True, returns a new stl_array instance.
        """
        
        # Check that O_Fourier is a bool
        if not isinstance(O_Fourier, bool):
            raise Exception("0_Fourier should be a bool")
    
        # If current status differs from desired
        if self.Fourier != O_Fourier:
            if O_Fourier:
                data = self.fourier_func(copy=copy)
            else:
                data = self.ifourier_func(copy=copy)
        # Already in desired space
        else:
            data = self.copy() if copy else self
    
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
        
        # Check that Fourier==False and dg=0
        if self.dg != 0:
            raise Exception("Mask is expected at a dg=0 resolution") 
        if self.Fourier:
            raise Exception("Mask is expected in real space") 
            
        # Pass the mask in a MR resolution.
        Mask_MR = StlData._init_MR(self.DT, None, self.N0,
                                    list(range(dg_max+1)), Fourier=False)
        _Mask_toMR = {"DT1": DT1_Mask_toMR, 
                      "DT2": DT2_Mask_toMR}.get(self.DT)
        Mask_MR.array = _Mask_toMR(self.array, self.N0, dg_max)
            
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
    
        if self.MR:
            raise ValueError("downsample requires MR == False input.")
    
        # Create new instance if copy
        data = self.copy(empty=True) if copy else self
    
        # Downsample using DT-specific backend
        subsampling_func = {
            "DT1": DT1_subsampling_func,
            "DT2": DT2_subsampling_func
        }.get(self.DT)
        data.array, data.Fourier = subsampling_func(
            self.array, self.Fourier, self.N0, self.dg, dg_out, mask_MR)
    
        # Update resolution and adjust Fourier status if needed
        data.dg = dg_out
        if O_Fourier is not None:
            data.out_fourier(O_Fourier)
    
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
    
        # Check that MR==False and dg=0
        if self.MR:
            raise ValueError("downsample_toMR requires MR == False input.")
        if self.dg != 0:
            raise Exception("Data are expected at a dg=0 resolution") 
            
        # Create new instance
        data = self.copy(empty=True)
    
        # Get DT-specific function to generate MR representation
        subsampling_func_toMR = {
            "DT1": DT1_subsampling_func_toMR,
            "DT2": DT2_subsampling_func_toMR
        }.get(self.DT)
        data.array, data.Fourier = subsampling_func_toMR(
            self.array, self.Fourier, self.N0, dg_max, mask_MR)
    
        # Update resolution and adjust Fourier status if needed
        data.MR = True
        data.dg = None
        data.list_dg = list(range(dg_max+1))
        if O_Fourier is not None:
            data.out_fourier(O_Fourier)
    
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
    
        if not self.MR:
            raise ValueError("downsample_fromMR requires MR == True input.")
    
        # Check that all listN resolutions are >= Nout
        for Nin in self.listN:
            if any(n_in < n_out for n_in, n_out in zip(Nin, Nout)):
                raise ValueError(
                    f"Cannot downsample from resolution {Nin} to {Nout}")
    
        # Get DT-specific backend
        subsampling_func_fromMR = {
            "DT1": DT1_subsampling_func_fromMR,
            "DT2": DT2_subsampling_func_fromMR
        }.get(self.DT)
    
        # Apply backend: result shape (..., len(listN), *Nout)
        array, Fourier = subsampling_func_fromMR(
            self.array, self.listN, Nout, self.Fourier)
    
        # Create output StlData (MR=False)
        data = StlData(array, self.DT, N=Nout, Fourier=Fourier)
        
        if O_Fourier is not None:
            data.out_fourier(O_Fourier)
    
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
        
        # Ensure data is in real space before computing modulus
        data = self.out_fourier(O_Fourier=False, copy=copy)
        
        # Get DT-specific backend
        modulus = {"DT1": DT1_modulus, "DT2": DT2_modulus}.get(self.DT)
        
        if data.MR:
            # Apply modulus to each array in the list
            data.array = [modulus(a) for a in data.array]
        else:
            # Single resolution case
            data.array = modulus(data.array)
                    
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
        
        # Check coherence of mask.
        if mask_MR is not None:
            if not isinstance(mask_MR, StlData):
                raise Exception("Mask should be a StlData instance") 
            if self.DT != mask_MR.DT:
                raise Exception("Mask and data should have same DT")
            if self.N0 != mask_MR.N0:
                raise Exception("Mask and data should have same N0")        
            if mask_MR.Fourier:
                raise Exception("Mask should be in real space") 
            if len(mask_MR.list_dg) < self.dg + 1:
                raise ValueError("Mask.list_dg should contain data dg value")
                
        if self.Fourier:
            data = self.ifourier_func(copy=True)
        else:
            data = self
            
        # Compute mean
        if data.MR is False:
            _mean_func = {"DT1": DT1_mean_func, \
                         "DT2": DT2_mean_func}.get(self.DT)
            mean = _mean_func(
                data.array, self.N0, self.dg, square,
                None if mask_MR is None else mask_MR[self.dg])
        else:
            _mean_func_MR = {"DT1": DT1_mean_func_MR, \
                         "DT2": DT2_mean_func_MR}.get(self.DT)
            mean = _mean_func_MR(
                data.array, self.N0, self.list_dg, square, mask_MR)
                    
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
        
        # Check DT consistency of data2 if not None
        data1 = self
        if data2 is not None:
            if not isinstance(data2, StlData):
                raise Exception("Data2 should be a StlData instance") 
            if data1.DT != data2.DT:
                raise Exception("Data1 and Data2 should have same DT")
            if data1.N0 != data2.N0:
                raise Exception("Data1 and Data2 should have same N0")
            if data1.dg != data2.dg:
                raise Exception("Data1 and Data2 should have same dg")            
        else:
            data2 = data1
            
        # Check coherence of mask.
        if mask_MR is not None:
            if not isinstance(mask_MR, StlData):
                raise Exception("Mask should be a StlData instance") 
            if data1.DT != mask_MR.DT:
                raise Exception("Mask and data should have same DT")
            if data1.N0 != mask_MR.N0:
                raise Exception("Mask and data should have same N0")        
            if mask_MR.Fourier:
                raise Exception("Mask should be in real space") 
            if len(mask_MR.list_dg) < data1.dg + 1:
                raise ValueError("Mask.list_dg should contain data dg value")
    
        # Compute covariance
        _cov_func = {
            "DT1": DT1_cov_func,
            "DT2": DT2_cov_func,
        }.get(data1.DT)
        cov = _cov_func(data1.array, data1.Fourier,
                    data2.array, data2.Fourier,
                    data1.N0, data1.dg,
                    None if mask_MR is None else mask_MR.array[data1.dg],
                    remove_mean
                    )
        
        return cov