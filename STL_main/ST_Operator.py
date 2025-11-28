# -*- coding: utf-8 -*-
"""
Main structure of STL

Tentative proposal by EA 
"""

import numpy as np
from ST_Statistics import ST_Statistics
import torch as bk # from_numpy, zeros, dim, shape, nan
                    
###############################################################################
###############################################################################

class ST_Operator:
    '''
    Class whose instances correspond to scattering transforms operators.
    The operator is built through __init__ method.
    The operator is applied through apply method.
    This operator is DT-independent, and call sub-functions with common 
    I/O structure, which in turn rely on DT-dependent backend.
    
    When the ST operator is applied to some data, it creates an instance of the 
    ST statistics where all necessary parameters are passed such that the ST 
    operator that was used in the computation can be reconstructed from it if 
    necessary.
    
    To allow that, a default setting for all parameters used for the apply
    method can be stored in the ST operator. 
    
    A prescription is also given on the order of which the different 
    normalizations/compression can be done:
        norm -> iso -> angular_ft -> scale_ft -> flatten (mask_st)
    Not every transform can be used, but the ordering should be respected.
    For instance:
        vanilla -> norm -> angular_ft -> flatten (mask_st)
        vanilla -> iso -> scale_ft 
    This allow the operators to be defined in a unique way from these 
    parameters.    
        
    Mask can be stored in the operator. 
    
    Parameters
    ----------
    # Data and Wavelet Transform
    - data : instance of some STL_Data_Class
        Data (1d, 2d planar, HealPix, 3d) ##################################################
    - J : int
        number of scales
    - L : int
        number of orientations
    - WType : str
        type of wavelets
        
    # Scattering Transform
    - SC : str 
        type of ST coefficients ("ScatCov", "WPH")
    - jmin : int
        minimum scale for ST statistics computation
    - jmax : int
        maximum scale for ST statistics computation
    - dj : int
        maximum scale difference for ST statistics computation
    - pbc : bool
        periodic boundary conditions
    - mask_MR : StlData with MR=True or None
        Multi-resolution masks, requires list_dg = range(dg_max + 1)
        
    # Additional transform/compression
    - norm : str
        type of norm ("S2", "S2_ref")
    - S2_ref : array
        array of reference S2 coefficients
    - iso : bool
        keep only isotropic coefficients
    - angular_ft : bool
        perform angular fourier transform on the ST statistics
    - scale_ft : bool
        perform scale cosine transform on the ST statistics
    - flatten : bool
        only return a 1D-array and not a ST_Statistics instance
    - mask_st : list of position
        mask to be applied when flatten ST statistics
    
    Attributes 
    ----------
    - parent parameters (see above)
    - wavelet_op : Wavelet_Transform class
        Wavelet Transform operator
    
    '''

    ########################################
    def __init__(self, 
                 data, J=None, L=None, WType=None,
                 SC="ScatCov", jmin=None, jmax=None, dj=None,
                 pbc=True, mask=None, 
                 norm="S2", S2_ref=None, iso=False, 
                 angular_ft=False, scale_ft=False,
                 flatten=False,
                 wavelet_array = None,
                 wavelet_array_MR = None,
                 dg_max = None,
                 j_to_dg = None,
                 Single_Kernel = None,
                 mask_st=None,
                 mask_opt = False,
                 mask_MR = None
                 ):
        '''
        Constructor, see details above.
        
        The mask parameter either expect:
            - a MR=False mask at dg=0, then converted to the mask_MR format.
            - a MR=True mask_MR with list_dg = range(dg_max + 1)
        '''
        # Main parameters
        self.DT = data.DT
        self.N0 = data.N0
        
        # Wavelet transform and related parameters
        self.wavelet_op = data.get_wavelet_op(J=J, L=L) #Wavelet_Operator(DT, N0, J, L, WType)
        self.J = self.wavelet_op.J
        self.L = self.wavelet_op.L
        self.WType = self.wavelet_op.WType
        self.dg = data.dg
        
        # Scattering transform related parameters
        self.SC = SC
        self.jmin = jmin
        self.jmax = jmax
        self.dj = dj
        self.pbc = pbc
        
        # Additional transform/compression related parameters
        self.norm = norm
        self.S2_ref = S2_ref
        self.iso = iso
        self.angular_ft = angular_ft
        self.scale_ft = scale_ft
        self.flatten = flatten
        self.mask_st = mask_st
        self.mask_MR = mask_MR
        
        # Check mask coherence and construct MR mask if necessary
        self.mask_MR = None
        if mask is not None:
            if not self.wavelet_op.mask_opt:
              raise Exception(
                  "Wavelet transform with masks not supported for this DT") 
            if not isinstance(mask, StlData):
                raise Exception("Mask should be a StlData instance") 
            if self.DT != mask.DT:
                raise Exception(
                    "Mask and wavelet transform should have same DT")
            if self.N0 != mask.N0:
                raise Exception(
                    "Mask and wavelet transform should have same N0")        
            if mask.Fourier:
                raise Exception("Mask should be in real space") 
            if not mask.MR:
                self.mask_MR = mask.downsample_toMR_Mask(
                    self.wavelet_op.dg_max)
            else:
                if mask.list_dg != list(range(self.wavelet_op.dg_max+1)):
                    raise Exception(
                    "Mask should be between MR between dg=0 and dg_max") 
                self.mask_MR = mask
        
    ########################################
    @classmethod
    def from_ST_Statistics(self, st_stat, N0_new=None):
        '''
        Alternative constructor, which generates the ST operator used to 
        compute a given set of ST statistics.
        
        Parameters
        ----------
        - st_stat : ST_Statistics
            st_stat instance whose parameters have to be reproduced
        - N0_new : tuple
            new initial size of array (can be multiple dimensions)
            
        Remark and to do
        ----------
        - In fact, a ST_Statistics instance cannot transmit the flatten 
        parameter, since it would have return a 1D array. This is not clear
        for me how to deal with this point.
            
        '''
        
        N0 = st_stat.N0 if N0_new is None else N0_new
        
        return ST_Operator(st_stat.DT, 
                           N0, 
                           J =st_stat.J, 
                           L=st_stat.L, 
                           WType=st_stat.WType, 
                           SC=st_stat.SC, 
                           jmin=st_stat.jmin,
                           jmax=st_stat.jmax,
                           dj=st_stat.dj,
                           pbc=st_stat.pbc, 
                           mask_MR=st_stat.mask_MR, 
                           norm=st_stat.norm, 
                           S2_ref=st_stat.S2_ref,
                           iso=st_stat.iso, 
                           angular_ft=st_stat.angular_ft,
                           scale_ft=st_stat.scale_ft,
                           flatten = st_stat.flatten,
                           mask_st=st_stat.mask_st
                           )
        
    ########################################
    def apply(self, data, 
              SC=None, jmin=None, jmax=None, dj=None,
              pbc=None, mask_MR=None, pass_mask=False,
              norm=None, S2_ref=None, iso=None, 
              angular_ft=None, scale_ft=None,
              flatten=None, mask_st=None):
        '''
        Compute the Scattering Transform (ST) of data, which are either stored 
        in an instance of the ST statistics class, or returned as a flatten
        array.
        
        This DT-independent methods calls sub-functions which have a common 
        I/O structure, and in turn rely on DT-dependent backend.
        
        It outputs an instance of the Scattering Statistics class, whose 
        additional methods can be called directly to get the desired output.
        
        Uses ST operator parameters unless explicitly overridden in apply.
        
        !!! Attention: I give an example in torch here, but we should consider
        how to include different backend !!!
        
        !!! Attention: I give here the version with standard scat cov !!!

        Parameters
        ----------
        # Data
        - data : StlData with MR=False, dim (N) or (Nc, N) or (Nb, Nc, N)
            data, Nc number of channel, Nb batch size. Should have dg=0.
    
        # Scattering Transform
        - SC : str 
            type of ST coefficients ("ScatCov", "WPH")
        - jmin : int
            minimum scale for ST statistics computation
        - jmax : int
            maximum scale for ST statistics computation
        - dj : int
            maximum scale difference for ST statistics computation
        - pbc : bool
            periodic boundary conditions
        - mask_MR : StlData with MR=True or None
            Multi-resolution masks, requires list_dg = range(dg_max + 1)
        - pass_mask : bool
            Pass mask to ST statistics object if True
            
        # Additional transform/compression
        - norm : str
            type of norm ("S2", "S2_ref")
        - S2_ref : array
            array of reference S2 coefficients
        - iso : bool
            keep only isotropic coefficients
        - angular_ft : bool
            perform angular fourier transform on the ST statistics
        - scale_ft : bool
            perform scale cosine transform on the ST statistics
        - flatten : bool
            only return a 1D-array and not a ST_Statistics instance
        - mask_st : list of position
            mask to be applied when flatten ST statistics
            
        Output 
        ----------
        - data_st : ST_Statistics instance, or 1D array
            ST statistics of I, as a flatten array if flatten=True        
        '''
        
        ########################################
        # General Initialization 
        ########################################
        
        # Consistency checks 
        if self.N0 != data.N0:
            raise Exception(
                "Scattering operator and data should have same N0") 
        if self.dg != data.dg:
            raise Exception("Data expected with dg=0") 
        
        # Local value for the wavelet transform parameters
        N0 = self.N0
        J = self.J
        L = self.L
        WType = self.wavelet_op.WType
        
        # Local value for the scattering transform parameters
        SC = self.SC if SC is None else SC
        jmin = self.jmin if jmin is None else jmin
        jmax = self.jmax if jmax is None else jmax
        dj = self.dj if dj is None else dj
        pbc = self.pbc if pbc is None else pbc
        mask_MR = self.mask_MR if mask_MR is None else mask_MR
        
        # Local value for the additional transforms parameters
        norm = self.norm if norm is None else norm
        S2_ref = self.S2_ref if S2_ref is None else S2_ref
        iso = self.iso if iso is None else iso
        angular_ft = self.angular_ft if angular_ft is None else angular_ft
        scale_ft = self.scale_ft if scale_ft is None else scale_ft
        flatten = self.flatten if flatten is None else flatten
        mask_st = self.mask_st if mask_st is None else mask_st
        
        # Put in torch or relevant bk
        if type(data.array) == np.ndarray:
            data.array = bk.from_numpy(data.array)
        
        # Put in the expected size: (Nb, Nc, N) 
        N_DT = len(N0)
        if data.array.dim() == N_DT:
            data.array = data.array[None, None, ...]                   #(1,1,N)
        elif data.array.dim() == N_DT + 1:
            data.array = data.array[None, ...]                        #(1,Nc,N)            
        Nb, Nc = data.array.shape[0],data.array.shape[1]
        
        # Create a ST_statistics instance
        data_st = ST_Statistics(self.DT, N0, J, L, WType,
                                SC, jmin, jmax, dj,
                                pbc, mask_MR if pass_mask else None,
                                Nb, Nc, self.wavelet_op)
            
        # Define the mask for conv computation if necessary
        if not pbc:
            mask_bc = self.construct_mask_bc(mask_MR)                   
        else: 
            mask_bc = mask_MR
            
        # Initialize ST statistics values
        # Add readability w.r.t. having it in the ST statistics initilization
        if self.SC == "ScatCov":
            data_st.S1 = bk.zeros((Nb,Nc,J,L))
            data_st.S2 = bk.zeros((Nb,Nc,J,L))
            data_st.S3 = bk.zeros((Nb,Nc,J,J,L,L),dtype=bk.complex128) + bk.nan
            data_st.S4 = bk.zeros((Nb,Nc,J,J,J,L,L,L),dtype=bk.complex128) + bk.nan
        
        ########################################
        # ST coefficients computation
        ########################################

        # I give here 4 different ST computations, for illustration purpose.
        # They are without npbc, masks, jmin, jmax, dj...
        # We assume that scales increase with j.
        # We only do a first convolution at all scales, without a MR framework.
        
        ### !! WARNING: simple code break here !! ###
        # Should probably be segmented in subfunction.
        
        ########################################
        ### Version vanilla ###
        
        # Vanilla version uses the following form for S3 and S4
        # S3 = Cov(|I*psi1|*psi2, I*psi2)
        # S4 = Cov(|I*psi1|*psi3, |I*psi2|*psi3)
        
        data_l1m={}
        l_data=data.copy()
        ### Higher order computation ###
        for j3 in range(J):
            # Compute first convolution and modulus
            
            data_l1 = self.wavelet_op.apply(l_data,j3)                  #(Nb,Nc,L,N3)
            data_l1m[j3] = data_l1.modulus(inplace=False)                #(Nb,Nc,L,N3) 
            
            # Compute S1 and S2
            data_st.S1[:,:,j3,:] = data_l1m[j3].mean()                          #(Nb,Nc,J,L)
            data_st.S2[:,:,j3,:] = data_l1m[j3].mean(square=True)               #(Nb,Nc,J,L)
             
            data_l1m_l2m={}
            for j2 in range(j3+1): 
                data_l1m_l2 = self.wavelet_op.apply(data_l1m[j2],j3) # (Nb,Nc,L2,L3,N3)
                # S3(j2,j3) = Cov(|I*psi2|*psi3, I*psi3) 
                data_st.S3[:,:,j2,j3,:,:] = data_l1m_l2[:,:,None].cov(          #(Nb,Nc, 1,L3,N3)
                                 data_l1                                        #(Nb,Nc, 1,L3,N3)
                                 )        
                                 
                data_l1m_l2m[j2] = data_l1m_l2.modulus(inplace=False)                #(Nb,Nc,L,N3)
                
                for j1 in range(j2+1):
                    # S4(j1,j2,j3) = Cov(|I*psi1|*psi3, |I*psi2|*psi3)
                    data_st.S4[:,:,j1,j2,j3,:,:,:] = data_l1m_l2m[j1][:,:,None,:].cov(
                            data_l1m_l2m[j2][:,:,:,None]
                            )         
                            
            # Downsample at Nj3
            l_data.downsample(j3+1)                  #(Nb,Nc,j3+1,L,N3)
            for j2 in range(j3+1): 
                data_l1m[j2].downsample(j3+1)                 #(Nb,Nc,j3+1,L,N3)
            
        ########################################
        # Additional transform/compression
        ########################################
        # Normalisation
        if norm is None:
            pass
        elif norm == "store_ref":
            if self.S2_ref is not None:
                print("S2_ref of the ST_Op is overwrote")
            data_st.to_norm(norm)
            self.S2_ref = data_st.S2_ref
        elif norm == "load_ref":
            if S2_ref is None:
                raise Exception("S2_ref should be stored in the ST_Operator")
            data_st.to_norm(norm, S2_ref = self.S2_ref)
            
        if iso:
            data_st.to_iso()
            
        if angular_ft:
            data_st.to_angular_ft()
            
        if scale_ft:
            data_st.to_scale_ft()            
            
        if flatten:
            data_st.flatten(mask_st)
        
        return data_st
    
    ########################################
    def construct_mask_bc(self, mask_MR):
        '''
        Construct the MR mask used when computing mean and covariances.
        It takes into acount the initial mask_MR if not None.
        Else, it just return a mask to deal with the non-periodic boundary 
        conditions.

        To be written.
        
        Returns
        -------
        mask_bc_MR : TYPE
            DESCRIPTION.

        '''

        
        return 1
        
    
