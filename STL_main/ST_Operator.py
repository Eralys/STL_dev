# -*- coding: utf-8 -*-
"""
Main structure of STL

Tentative proposal by EA 
"""

import torch
import numpy as np
from .WaveletTransform import stl_data, Wavelet_Operator
from .ST_Statistics import ST_Statistics
                    
###############################################################################
###############################################################################

class ST_Operator:
    '''
    Class whose instances correspond to an operator to perform scattering transforms.
    The operator is built through __init__ method.
    The operator is applied through apply method.
    This operator is DT-independent, and call sub-functions which have a common 
    I/O structure, which in turn rely on DT-dependent backend.
    
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
    - WType : str
        type of wavelets
    - SC : str ("ScatCov", "WPH")
        type of scattering coefficients to be computed
    
    Attributes 
    ----------
    - parent parameters (DT,N,J,L,WType,SC)
    - wavelet_op : Wavelet_Transform class
        Operator for Wavelet Transform
        Also contain the sub-sampling method
    
    Questions and to do
    ----------
    '''

    ########################################
    def __init__(self, DT, N, J=None, L=None, WType=None, SC="ScatCov"):
        '''
        Constructor, see details above.
        '''
        # Main parameters
        self.DT = DT
        self.N = N
        self.wavelet_op = Wavelet_Operator(DT, N, J, L, WType)
        
        # Set J, L, WType to standard values if None
        self.J = self.wavelet_op.J
        self.L = self.wavelet_op.L
        self.WType = self.wavelet_op.WType
        
    ########################################
    def apply(self, data, jmin=None, jmax=None, dj=None, \
                  pbc=True, mask=None): # + submethods options.
        '''
        Compute the Scattering Transform (ST) from the stl_data data.
        This method is DT-independent, and call sub-functions which have a common 
        I/O structure, which in turn rely on DT-dependent backend.
        It outputs an instance of the Scattering Statistics class.
        Additional sub-methods can be called directly to get the desired output
        
        !!! Attention: I give an example in torch here, but we should consider
        how to include different backend !!!
        
        !!! Attention: I give here the version with standard scat cov !!!

        Parameters
        ----------
        - data : stl_data, dim (N) or (Nc, N) or (Nb, Nc, N)
            data, can be batched on several dimensions
            data propose the form above with Nc the number of channels and
            Nb the batch size
        - I_fourier : bool
            is the input image in real space or in Fourier space.
        - jmin : int
            minimum scale at which the ST statistics are computed
        - jmax : int
            maximum scale at which the ST statistics are computed
        - dj : int
            maximum scale ratio at which the ST statistics are computed
        - pbc : bool
            periodic boundary conditions
        - mask : (..., N) or (N)
            mask(s) to be used during the convolution. Either a single mask
            or a collection of masks can be given (should match some of the
            last dimensions of I if it is the case)
            
        Output 
        ----------
        - I_st : scattering statistics instance
            ST statistics of I
            -> output can be refined by direct application of submethod.
            
        '''
        
        ########################################
        # General Initialization 
        ########################################
        
        # Consistency checks
        if self.DT != data.DT:
            raise Exception("Scattering operator and data should have same data type") 
        if self.N != data.N:
            raise Exception("Scattering operator and data should have same data size") 
            # I prefer to impose that to avoid unefficient computation. 
        
        # Shortcut for better readibility
        N = self.N
        J = self.J
        L = self.L
        Nj = self.wavelet_op.MR_array
        
        # Initialize parameter values if None
        jmin = 0 if jmin==None else jmin
        jmax = self.J if jmax==None else jmax
        dj = self.J if dj==None else dj
        
        # Create a ST_statistics instance
        data_st = ST_Statistics(self.N, self.J, self.L, self.WType, self.SC, \
                             jmin, jmax, dj)
            
        # Identify the dimension of the image, reshape, and pass in torch
        # Put in the expected size: (Nb, Nc, N) 
        N_DT = len(N)
        if data.data.ndim == N_DT:
            data.data = data.data[None, None, ...]                     #(1,1,N)
        elif data.data.ndim == N_DT + 1:
            data.data = data.data[None, ...]                          #(1,Nc,N)            
        Nb, Nc = data.data.shape[0], data.data.shape[1]
        # Put in torch <- to see with backend
        if type(data.data) == np.ndarray:
            data.data = torch.from_numpy(data.data)
            
        # Define the mask for conv computation if necessary
        if not pbc:
            mask_bc = self.construct_mask_bc()
            mask_bc = mask_bc[None, None, ...]                         #(1,1,N)     
        else: 
            mask_bc = None
            
        # Initialize ST statistics values
        # Could be added in the data_st initialization
        data_st.S1 = torch.zeros((Nb,Nc,J,L))
        data_st.S2 = torch.zeros((Nb,Nc,J,L))
        data_st.S3 = torch.zeros((Nb,Nc,J,J,L,L)) + np.nan
        data_st.S4 = torch.zeros((Nb,Nc,J,J,J,L,L)) + np.nan
        
        ########################################
        # First order computation
        ########################################
        
        # First version, simple form and without npbc, masks, jmin, jmax, dj...
        # Here scales increase with j. 
        
        # Attention !! Here at all scales, and without a MR framework, for the
        # first convolution. I didn't code the other version for the moment.
        if self.wavelet_op.full_conv:
        
            # Compute first convolution and modulus
            data_l1 = self.wavelet_op.apply(data)                #(Nb,Nc,J,L,N)
            data_l1m = data_l1.module_fonc(copy=True)            #(Nb,Nc,J,L,N) 
            # Replacement if we do not need data_l1
            data_l1m = self.wavelet_op.apply(data).module_fonc() #(Nb,Nc,J,L,N) 
            
            # Compute S1 and S2
            data_st.S1 = data_l1m.mean_fonc()                      #(Nb,Nc,J,L)
            data_st.S2 = data_l1m.mean_fonc(square=True)           #(Nb,Nc,J,L)
        
        ########################################
        # Higher order computation
        # Version vanilla 
        ########################################
        
        for j3 in range(J):
            # Scale smaller-eq to j3 whose [I*psi| will be convolved at j3
            data_l1.data = data_l1.data[:,:,:j3+1]            #(Nb,Nc,j3+1,L,N)
            data_l1m.data = data_l1m.data[:,:,:j3+1]          #(Nb,Nc,j3+1,L,N)
            # Batch downsample at Nj3
            data_l1.downsample(Nj[j3])                      #(Nb,Nc,j3+1,L,Nj3)
            data_l1m.downsample(Nj[j3])                     #(Nb,Nc,j3+1,L,Nj3)
            
            # Compute |I*psi2|*psi3                      #(Nb,Nc,j3+1,L2,L3,N3)
            data_l1m_l2 = self.wavelet_op.apply(data_l1m, j3) 
            
            for j2 in range(j3+1):
                # S3(j2,j3) = Cov(|I*psi2|*psi3, I*psi3) 
                data_st.S3[:,:,j2,j3,:,:] = stl_data.cov(
                                 data_l1m_l2[:,:,j2],         #(Nb,Nc,L2,L3,N3)
                                 data_l1[:,:,j3,None]         #(Nb,Nc, 1,L3,N3)
                                 )        
                
                for j1 in range(j2+1):
                    # S4(j1,j2,j3) = Cov(|I*psi1|*psi3, |I*psi2|*psi3)
                    data_st.S4[:,:,j1,j2,j3,:,:,:] = stl_data.cov(
                            data_l1m_l2[:,:,j1,:,None],    #(Nb,Nc,L1, 1,L3,N3)
                            data_l1m_l2[:,:,j2,None,:]     #(Nb,Nc, 1,L2,L3,N3)
                            )
                
        ########################################
        # Higher order computation
        # Version batchée vanilla 
        ########################################
        
        for j3 in range(J):
            # Scale smaller-eq to j3 whose [I*psi| will be convolved at j3
            data_l1.data = data_l1.data[:,:,:j3+1]            #(Nb,Nc,j3+1,L,N)
            data_l1m.data = data_l1m.data[:,:,:j3+1]          #(Nb,Nc,j3+1,L,N)
            # Batch downsample at Nj3
            data_l1.downsample(Nj[j3])                      #(Nb,Nc,j3+1,L,Nj3)
            data_l1m.downsample(Nj[j3])                     #(Nb,Nc,j3+1,L,Nj3)
            
            # Compute |I*psi2|*psi3                      #(Nb,Nc,j3+1,L2,L3,N3)
            data_l1m_l2 = self.wavelet_op.apply(data_l1m, j3) 
            
            # S3(j2,j3) = Cov(|I*psi2|*psi3, I*psi3) 
            data_st.S3[:,:,:j3+1,j3,:,:] =stl_data.cov(
                       data_l1m_l2[:,:,:j3+1],           #(Nb,Nc,j3+1,L2,L3,N3)
                       data_l1[:,:,j3,None,None]         #(Nb,Nc,   1, 1,L3,N3)
                       )
                
            for j2 in range(j2+1):
                # S4(j1,j2,j3) = Cov(|I*psi1|*psi3, |I*psi2|*psi3)
                data_st.S4[:,:,:j2+1,j2,j3,:,:,:] = stl_data.cov(
                    data_l1m_l2[:,:,:j2+1,:,None],    #(Nb,Nc,j2+1,L1, 1,L3,N3)
                    data_l1m_l2[:,:,j2,None,None,:]   #(Nb,Nc,   1, 1,L2,L3,N3)
                        )
         
        ########################################
        # Higher order computation
        # Version Sihao (adaptée)
        ########################################
        
        # -> no need to store data_l1 at all !
        
        for j3 in range(J):
            # Scale smaller-eq to j3 whose [I*psi| will be convolved at j3
            data_l1m.data = data_l1m.data[:,:,:j3+1]          #(Nb,Nc,j3+1,L,N)
            # Batch downsample at Nj3
            data_l1m.downsample(Nj[j3])                      #(Nb,Nc,j3+1,L,N3)
            
            for j2 in range(j3+1):
                # Compute |I*psi2|*psi3                       #(Nb,Nc,L2,L3,N3)
                data_l1m_l2 = self.wavelet_op.apply(data_l1m[:,:,j2], j3) 
                
                # S3(j2,j3) = Cov(I, |I*psi2|*psi3) 
                data_st.S3[:,:,j2,j3,:,:] = stl_data.cov(
                                 data[:,:,None,None],         #(Nb,Nc, 1, 1,N)
                                 data_l1m_l2[:,:]             #(Nb,Nc,L2,L3,N3)
                                 )        
                
                for j1 in range(j2+1):
                    # S4(j1,j2,j3) = Cov(|I*psi1|, |I*psi2|*psi3)
                    data_st.S4[:,:,j1,j2,j3,:,:,:] = stl_data.cov(
                         data_l1m[:,:,j1,:,None,None],     #(Nb,Nc,L1, 1, 1,N3)
                         data_l1m_l2[:,:,None,:,:]         #(Nb,Nc, 1,L2,L3,N3)
                            )
        
        ########################################
        # Higher order computation
        # Version Sihao batchée
        ########################################
        
        # -> no need to store data_l1 at all !
        
        for j3 in range(J):
            # Scale smaller-eq to j3 whose [I*psi| will be convolved at j3
            data_l1m.data = data_l1m.data[:,:,:j3+1]         #(Nb,Nc,:j3+1,L,N)
            # Batch downsample at Nj3
            data_l1m.downsample(Nj[j3])                    #(Nb,Nc,:j3+1,L,Nj3)
            
            # Compute |I*psi2|*psi3                     #(Nb,Nc,:j3+1,L2,L3,N3)
            data_l1m_l2 = self.wavelet_op.apply(data_l1m, j3) 
                        
            # S3(j2,j3) = Cov(|I*psi2|*psi3, I*psi3) 
            data_st.S3[:,:,:j3+1,j3,:,:] =stl_data.cov(
                       data[:,:,None,None,None],         #(Nb,Nc,   1, 1, 1,N)
                       data_l1m_l2[:,:,:j3+1]            #(Nb,Nc,j3+1,L2,L3,N3)
                       )
                
            for j2 in range(j3+1):
                    # S4(j1,j2,j3) = Cov(|I*psi1|, |I*psi2|*psi3)
                data_st.S4[:,:,:j2+1,j2,j3,:,:,:] = stl_data.cov(
                  data_l1m[:,:,:j2+1,:,None,None],    #(Nb,Nc,j2+1,L1  1, 1,N3)
                  data_l1m_l2[:,:,j2,None,None],      #(Nb,Nc,   1, 1,L2,L3,N3)
                        )
        
        return data_st
    
    ########################################
    def construct_mask_bc(self):
        '''
        Construct the mask used when computing covariances when pbc==False

        Returns
        -------
        mask_bc : TYPE
            DESCRIPTION.

        '''
        
        # Need to be done
        mask_bc = np.ones(self.N)
        
        return mask_bc
        
    