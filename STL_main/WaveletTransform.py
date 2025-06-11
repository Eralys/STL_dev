# -*- coding: utf-8 -*-
"""
Main structure of STL

Tentative proposal by EA 

Remarques préliminaires :
    - Il faudrait séparer les fonctions qui sont data-type dépendantes (constru-
      ction des wavelets,wavelet transform, subsampling) de celles qui ne le 
      sont pas (calcul des ST statistics et ce qui en découle).
    - Autant que possible, il faudrait que les fonctions fixent automatiquement
      des choix standards pour tous les paramètres (e.g., J = log2(shape) – 2),
      pour que l’utilisation soit la plus simple possible pour un profane.
    - Ce n'est pas forcément grave a priori si toutes les fonctionnalités ne 
      sont pas accessibles sur tous les types de données (comme d'utiliser des
      masques, ou bien de faire des convolutions à toutes les échelles et sans
      multi-résolution pour les convolutions directes en HealPix).
    - En plus de la structure générale du code, il faudra prévoir les fonctions
      toutes intégrées (comme juste faire une synthèse à partir d'une carte), 
      ainsi que les notebooks d'exemple.
    - Dans tout le repo, je parle de real space et Fourier space. Pour des 
      cartes sphériques, cela correspond à l'espace des harmoniques sphériques.
    - Je propose de générer par défaut la banque d'ondelettes avec toutes les 
      échelles, et de ne sélectionner des éventuels jmin et jmax que lors du 
      calcul des ST statistics.
     
To do:
    - voir comment gérer différents backends de langages (torch/jax/tf...)
        -> cela va jouer un rôle dans les appels aux optimizers.
        -> de même pour cpu/gpu
    - je propose de travailler uniquement avec des données réelles
"""

from DataType1 import DT1_wavelet_build, DT1_wavelet_conv, DT1_cov_fonc,\
                      DT1_subsampling_fonc, DT1_fourier, DT1_ifourier,\
                      DT1_mean_fonc, DT1_findN, DT1_WT_prop,\
                      DT1_wavelet_build_j, DT1_wavelet_conv_full
from DataType2 import DT2_wavelet_build, DT2_wavelet_conv, DT2_cov_fonc,\
                      DT2_subsampling_fonc, DT2_fourier, DT2_ifourier,\
                      DT2_mean_fonc, DT2_findN, DT2_WT_prop
         
###############################################################################
###############################################################################
class stl_data:
    '''
    Class which contain the different types of data used in STL.
    Allow to store important parameters, such as DT, N, and the Fourier type.
    Also allow to convert from numpy to JAX (or other type).
    Allow to transfer internally these parameters.
    
    Has different standard functions as methods (fft, ifft, mean, module, 
    subsample, upsample, ...)
    
    Can be of any shape, as long as the last dimension correspond to the 
    tuple N.
    
    Parameters
    ----------
    - data : array (..., N)
        data to store
    - DT : str
        Type of data (1d, 2d planar, HealPix, 3d)
    - N : tuple
        size of the data (can be multiple dimensions)
    - Fourier : bool
         if in Fourier
    - type : str
        "npy", "jax", ...
    
    Attributes 
    ----------
    - parent parameters (data, DT, N, Fourier, type)
    '''
    
    ###########################################################################
    def __init__(self, data, DT, N=None, Fourier=False):
        '''
        Constructor, see details above.
        If initialize with data=None, N should be given.
        '''
        # Main parameters
        self.data = data
        self.DT = DT
        self.N = N
        self.Fourier = Fourier
        
        
        # find N from data if N==None, DT-dependent
        if N is None:
            findN = {"DT1": DT1_findN, "DT2": DT2_findN}.get(self.DT)
            self.N = findN(self.data, self.Fourier)
        
        self.build()
        
    ###########################################################################
    def copy(self, empty=False, N=None):
        """
        Copy a stl_data instance.
        
        Parameters
        ----------
        empty : bool
            If True, set data to None.
        Nj : int or None
            New resolution to override self.N if provided.
        """
        
        new_data = None if empty else self.data
        new_N = N if N is not None else self.N
        
        return stl_data(new_data, self.DT, new_N, self.Fourier)

    ###########################################################################
    def __getitem__(self, key):
        """
        To slice directly the data attribute.
        
        Default setting is a copy, to match with usual practices.
        This allows to write:
            object = stl_data(rand(4, 5, 6, 7), DT=DT, N=N, Fourier=Fourier)
            obj2 = obj[:, :, 3, :]
            obj3 = obj[:2]
        with usual slicing
        
        To modify directly self.data, one can simply do
        obj.data = obj.data[:,:,:3]
        
        Parameters
        ----------
        key : slicing
            slicing of the data attribute
        """
        
        # Convert to tuple if it's a single item
        if not isinstance(key, tuple):
            key = (key,)

        # Create the stl array and insert the sliced data
        data = self.copy(empty=True)
        data.data = self.data[key]

        return data
    
    ###########################################################################
    def fourier_fonc(self, copy=False):
        '''
        Compute the fourier transform of data.
        Should be understood in a general way, would be an harmonic transform 
        for spherical data.
        
        Parameters
        ----------
        - copy : bool
            Create a new stl_data instance if copy=True
        '''    
        
        if self.Fourier:
          raise Exception("data already in Fourier") 
          
        # Create new instance if copy
        if copy:
            data = self.copy(empty=True)
        else:
            data = self
        
        # Perform Fourier transform, DT-dependent
        fourier = {"DT1": DT1_fourier, "DT2": DT2_fourier}.get(self.DT)
        data.data = fourier(self.data, self.N)

        # Update Fourier status
        data.Fourier = True
        
        return data #self if copy=False
            
    ###########################################################################

    def ifourier_fonc(self, copy=False):
        '''
        Compute the inverse fourier transform of data.
        Should be understood in a general way, would be an harmonic transform 
        for spherical data.
        
        Parameters
        ----------
        - copy : bool
            Create a new stl_data instance if copy=True
        '''    

        if self.Fourier==False:
          raise Exception("data already in real space") 
          
        # Create new instance if copy
        if copy:
            data = self.copy(empty=True)
        else:
            data = self
            
        # Perform inverse Fourier transform, DT-dependent
        ifourier = {"DT1": DT1_ifourier, "DT2": DT2_ifourier}.get(self.DT)
        data.data = ifourier(self.data, self.N)
        
        # Update Fourier status
        data.Fourier = False

        return data #self if copy=False
        
    ###########################################################################

    def out_fourier(self, O_Fourier, copy=False):
        '''
        Put the stl_data in O_fourier, if it is not already.
        
        Parameters
        ----------
        - copy : bool
            Create a new stl_data instance if copy=True
        '''    
    
        if self.Fourier and not O_Fourier:
            data = self.fourier_fonc(copy=copy)
        elif not self.Fourier and O_Fourier:
            data = self.ifourier_fonc(copy=copy)
        else:
            data = self.copy() if copy else self

        return data #self if copy=False

    ###########################################################################

    def module_fonc(self, copy=False):
        '''
        Compute the modulus of data.
        Pass in real space if data in Fourier.
        
        Parameters
        ----------
        - copy : bool
            Create a new stl_data instance if copy=True
        
        Remark 
        ----------
        - In terms of optimization, it could be worth to perform the ift
        to self for subsequent operations.
        '''    
        
        # Inverse Fourier transform is necessary.
        data = self.out_fourier(False, copy=copy)
        # Modulus (need proper backend)
        data.data = abs(data.data)
            
        return data #self if copy=False

    ###########################################################################

    def downsample(self, Nout, O_Fourier=None, copy=False):
        '''
        Pass the data at the Nout resolution.
        The specific Fourier status of the output can be given.
        Per default, it will be the one that requires the less computation,
        which is DT-dependent.
                
        IMPORTANT : This downsampling function could be quite complex in order 
        to deal with incomplete data. It needs to be thought through carefully.
        We need also to see how it communicates with masks, which could 
        potentially be called here.
        
        Parameters
        ----------
        - Nout : tuple
            size of the output data (can be multiple dimensions)
        - O_fourier : bool
            output data in Fourier space (True) or in Fourier space (True)
        - copy : bool
            Create a new stl_data instance if copy=True
        '''    
        
        # Create new instance if copy
        if copy:
            data = self.copy(empty=True)
        else:
            data = self
        
        # Downsample, DT-dependent
        subsampling_fonc = {"DT1": DT1_subsampling_fonc, \
                            "DT2": DT2_subsampling_fonc}.get(self.DT)
        data.data, data.Fourier = subsampling_fonc(self.data,  \
                                  self.N, Nout, self.Fourier, O_Fourier)
                    
        # Save new resolution and transform to correct Fourier status if asked
        data.N = Nout
        data.out_fourier(O_Fourier)
            
        return data #self if copy=False
    
    ###########################################################################

    def upsample(self, Nout, O_Fourier=None):
        '''
        Do we want this function?
        '''
        
    ###########################################################################

    def mean_fonc(self, square=False, mask=None):
        '''
        Compute the mean of data on the tuple N last dimensions.
        Mean is computed in real-space, so ift is done if necessary.
        No mask is used if mask == None.
        
        Quadratic mean |x|^2 if square = True
       
        Mask can be either a single mask (dim N) or a collection of masks.
        Need to think about how to deal with that.

        Parameters
        ----------
        - square : bool
            True if quadratic mean
        - mask : array (..., N) or (N)
            mask(s) to be used when computing the mean. 
            
        Output 
        ----------
        - mean : array (...)  
            mean of data in dim N
        '''    
            
        # Compute mean
        _mean_fonc = {"DT1": DT1_mean_fonc, \
                     "DT2": DT2_mean_fonc}.get(self.DT)
        mean = _mean_fonc(self.data, self.Fourier, square, mask)
            
        return mean
        
    ###########################################################################
        
    def cov_fonc(self, data2=None, mask=None):
        '''
        Compute the covariance between data1=self and data2 on the tuple N last
        dimensions. 
        -> If both data are not at the same resolution (N), they will
        be put at a Nout resolution corresponding to the lowest one. 
        -> Both data do not need to have the same Fourier status.
        
        Depending of the DT, covariances can be done either in real space, or 
        in Fourier space, or in both (for instance using Plancherel-Parseval
        for 2d planar data).
        This function either directly compute the covariance, or transform one 
        or both of the field in order to compute it (this is DT-dependent)
        Note that with a mask, the covariance necessarily need to be computed 
        in real space.
        
        Mask can be either a single mask (dim N) or a collection of masks.
        Need to think about how to deal with that.
        
        Parameters
        ----------
        - data2 : stl_data class
            stl_data
        - mask : (..., N) or (N)
                mask(s) to be used when computing the mean. 
            
        Output 
        ----------
        - cov : array (...)  
            mean of data1 and data2 in dim N
        '''    
        
        # for readability 
        data1 = self
        
        if data2==None:
            data2 = data1
            
        if not data1.DT == data2.DT:
          raise Exception("data should be of same DT") 
        
        # Put at the lower resolution
        if data1.N[-1] > data2.N[-1]:
            data1 = data1.downsample(data2.N, copy=True)
        elif data2.N[-1] > data1.N[-1]:
            data2 = data2.downsample(data1.N, copy=True)
            
        # Compute covariance
        _cov_fonc = {"DT1": DT1_cov_fonc, \
                     "DT2": DT2_cov_fonc}.get(self.DT)
        cov = _cov_fonc(data1.data, data1.Fourier, \
                        data2.data, data2.Fourier, mask)
            
        return cov

###############################################################################
###############################################################################
    
class Wavelet_Operator:
    '''
    Class whose instances correspond to an operator to perform wavelet transforms.
    The wavelet set and the operator is built during the initilization.
    The operator is applied through apply method.
    This method is data-type dependent, and actually calls independent iterations,
    but with common method and attribute structure.
    
    IMPORTANT: The wavelet array is DT-dependent 
        -> it is either defined in Fourier or in real-space
        -> for a given j, it either call for the same wavelet set, of different
        wavelet sets at different resolution
    
    IMPORTANT: The wavelet convolution is DT-dependent
        -> it could either be be performed in real or Fourier space

    Parameters
    ----------
    - DT : str
        Type of data (1d, 2d planar, HealPix, 3d)
    - N : tuple
        size of the data (can be multiple dimensions)
    - J : int
        number of scales
    - L : int
        number of orientations
        -> should have default values, e.g., J=log2(size)-2 and L=4 for 2d planar
    - WType : str
        type of wavelets
        -> e.g., "Morlet" or "Bump-Steerable"
    
    Attributes 
    ----------
    - parent parameters (DT,N,J,L,WType)
    - wavelet_array : array
        array of wavelets at all oriented scales at full resolution
        -> the exact form of the array is DT-dependent
    - MR_list : list of length (J) of Nj tuples
        list of resolution at each scale
        -> resolution depends on j-scale only
        -> it is DT- and WType-dependent
        -> should allow to perform all sub-sampling once wavelet set is created
    - full_conv : bool
        is it possible to do a convolution at all scales at the initial 
        resolution
    - wave_j : bool
        True if convolution at a given j only need a specific part of wavelet
        array (a priori at a Nj resolution)
    - mask_opt : bool
        is it possible to do use masked during the convolution
    - _wavelet_cache : dictionnary
        Will be used to store the wavelets at different scales at different
        resolution if necessary
        
    Questions and to do
    ----------
    - I propose not to let the user to play with the Fourier type of the
    wavelet, since the convolution will expect a given type. However, we could
    have a plot function that allows to plot in both space if this make sense.
    - Do we keep a low-pass filter per default, in addition to high-pass ones?
    - Need to fix j/l notations, and the conventions with scales
      -> j increasing with scales in real space?
    - Need to know if we want to give a possibility to have more fine tuning
      in the wavelet set.
      -> I propose to only have dyadic wavelets for the "main set", even if 
         the 'wavelets'' for the POO' could be fine tuned.
    - Need to think how to include an additional wavelet set for P00'. A priori,
      could be generated here and kept in an additional attribute. However, the
      way to compute the P00' is not necessarily similar than the wavelet 
      convolution.
    - for __ini__, we could ask either for DT and N, or for an example stl_data
      nstance, from which DT and N are obtained. Could be added if useful.
    '''

    ###########################################################################
    def __init__(self, DT, N, J=None, L=None, WType=None):
        '''
        Constructor, see details above.
        '''
        # Main parameters
        self.DT = DT
        self.N = N
        self.J = J
        self.L = L
        self.WType = WType
        self.wavelet_array = None
        self.MR_array = None
    
        # Build, fix J, L, and WType values if None
        self.build()
        
        # Set properties of the wavelet transform
        WT_prop = {"DT1": DT1_WT_prop, "DT2": DT2_WT_prop}.get(self.DT)
        self.full_conv, self.wave_j, self.mask_opt = WT_prop(self.DT)
        
    ###########################################################################
    def build(self):
        '''
        Build wavelet set and subsampling_factors, see details above.
        The standard values for J, L, and WType are also fixed if None.
        '''
        
        # Built the wavelet set
        wavelet_build = {"DT1": DT1_wavelet_build, \
                     "DT2": DT2_wavelet_build}.get(self.DT)
        self.wavelet_array, self.MR_array, self.J, self.L, self.WType \
            = wavelet_build(self.DT, self.N, self.J, self.L, self.WType)
            
    ###########################################################################
    def wavelet_j(self, j):
        """
        Build, store, or load the necessary wavelets at scale j.
    
        For some DTs (e.g., HealPix), this always return self.wavelet_array.
        In this case, self.wave_j is False.
        For others, it vary with resolution or scale.
        """
    
        # If always return wavelet_array
        if not self.wave_j:
            return self.wavelet_array
    
        # For variable wavelet DTs, use cache and builder
        if not hasattr(self, "_wavelet_cache"):
            self._wavelet_cache = {}
    
        # Return if already computed
        if j in self._wavelet_cache:
            return self._wavelet_cache[j]
    
        # Else ompute and save the wavelet array at scale j (not for all DT)
        wavelet_build_j = {"DT1": DT1_wavelet_build_j}.get(self.DT) 
        wavelet_array_j = wavelet_build_j(self.wavelet_array, self.MR_array, j)
        self._wavelet_cache[j] = wavelet_array_j
    
        return wavelet_array_j
            
    ###########################################################################
    def plot(self, Fourier=None):
        '''
        Plot the set (or a subset) of wavelets, either in Fourier or real space.
        Can add a selection of (j,l).
        '''
    
        # To be done
        
    ###########################################################################
    def apply(self, data, j=None, mask=None, O_Fourier=None):
                  
        '''
        Compute the Wavelet Transform (WT) of the image I.
        This method is data-type dependent, and actually calls independent 
        iterations, but with common method and attribute structure.
        
        The WT can be computed:
            - either at all scales and angle, but without a multi-resolution 
            output, which is possible only if self.full_conv == True.
            - either at a given scale j, with a multi-resolution framework.
        -> I propose not to let other possibilities, like at a given scale but 
           not in a multi-resolution framework, or a given (j,l), since the 
           library should be ST-centred. Indeed, this allows to simplify the 
           functions, the parametrization, and the I/O formats.
           
        Mask can be either a single mask (dim N) or a collection of masks.
        Need to think about how to deal with that.

        IMPORTANT:
        The convolution at all scales at the initial resolution is not possible
        for instance with direct convolution in healpix. On the other hand, it 
        can be numerically efficient to compute directly all convolution with 
        a FFT with 2D planar data. 
            -> I prefer to have a general structure even if some options do not
               work with some DT. 
            -> We should nevertheless be very careful that we want to rely on
               an ensemble of commands for the ST computations that should work
               with all DT.
               
        IMPORTANT:
        I propose not to deal with the issue of non-periodicity here, but 
        rather when computing mean and covariances. 
        There is however the possibility to use masks, for some DT, if 
        self.mask_opt == True.
        
        IMPORTANT:
        When doing the resolution at a given j scale in a MR framework, I 
        propose that the data can be given at any resolution N > Nj. The 
        downsampling at resolution Nj will then be done automatically, before
        to perform the convolution. 
        
        IMPORTANT: 
        We shouldn't fetch the entire wavelet set when a convolution a single 
        scale is done, but only send the necessarily data.
        
        Parameters
        ----------
        - data : stl_data, with data of dim (..., N)
            stl data, can be batched on several dimensions
        - j : int
            scale at which the convolution is done. Done at all scales if None.
        - mask : (..., N) or (N)
                mask(s) to be used when computing the mean. 
        - O_fourier : bool
            are the output data in real space or in Fourier space.
            Per default, it will be the one that requires the less computation,
            which is DT-dependent.
            
        Output 
        ----------
        - WT : stl_data, of dim:
            -> (..., J, L, N) if j == None
            -> (..., L, Nj) if j == int 
            Wavelet convolutions at different scales and angles. 
            
        Questions and to do
        ----------
        - The mask is only binary here, no?
        - mask should probably be put as stl_data objects, in order to be 
        downsampled easily.
        - We could think at the possibility to compute WT at fixed (j,l)
        values, if it helps distributing the computations for large batchs.
        '''
        
        # Same DT
        if self.DT != data.DT:
            raise Exception("Wavelet transform and data should have same data type") 
        
        # Are the parameters compatible with the DT.
        if j is None and not self.full_conv:
          raise Exception("Wavelet transform needs to be done at a specific scale j") 
        if mask is not None and not self.mask_opt:
          raise Exception("Wavelet transform with masks not supported for this data type") 
        
        # Convolution at all scales at the same time
        if j is None:
            # I propose the self.N == data.N constraint in full conv
            if self.N != data.N:
                raise Exception("Wavelet transform and data should have same resolution") 
            # Create the autput stl.data instance for the Wavelet Transform
            WT = data.copy(empty=True)
            # Compute the WT, all DT are not necessarily included here.
            wavelet_conv_full = {"DT1": DT1_wavelet_conv_full}.get(self.DT)
            WT.data, WT.Fourier = wavelet_conv_full(self.wavelet_set, \
                    data.data, data.Fourier, mask)
            
        # Convolution at a given scale in MR.
        else:
            if j is not int:
                raise Exception("Expect a single integer scale") 
            # Identify the Nj resolution
            Nj = self.MR_array[j]
            # Create the autput stl.data instance for the Wavelet Transform
            WT = data.copy(empty=True, N=Nj)
            # Compute the WT at a given j
            wavelet_conv = {"DT1": DT1_wavelet_conv, \
                     "DT2": DT2_wavelet_conv}.get(self.DT)
            WT.data, WT.Fourier = wavelet_conv(self.wavelet_j(j),\
                data.downsample(Nj, copy=True).data, data.Fourier, mask)
            # in practice, we need to downsample mask at Nj. To do later
        
        # Transform to correct Fourier status if necessary
        WT.out_fourier(O_Fourier)

        return WT
    
    ###########################################################################
    def downsample_j(self, data, j, O_fourier=None, copy=False):
        '''
        Transform a stl_data oject to put it a the Nj resolution.
        
        Parameters
        ----------
        - I : stl array
            data, can be batched on several dimensions
        - j : int
            scale defining the output resolution
        - O_fourier : bool
            is the output data in real space or in Fourier space.
            Per default, it will be the one that requires the less computation,
            which is DT-dependent.
        '''
        
        # downsample data at Nj resolution
        return data.downsample(self.MR_array[j], O_Fourier=O_fourier, copy=copy)

