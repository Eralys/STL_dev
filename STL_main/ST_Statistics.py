# -*- coding: utf-8 -*-
"""
Main structure of STL

Tentative proposal by EA 
"""

import backend as bk # mean, zeros

                    
###############################################################################
###############################################################################

class ST_Statistics:
    '''
    Class whose instances correspond to an set of scattering statistics
    The set of statistics is built by the ST_operator method, which use the 
    __init__ method. This class is DT-independent.
    
    This class contains methods that allow to deal with ST statistics in an 
    unified manner. Most of these methods can be applied directly through the 
    ST_operator implementation.
    
    When used in loss, a 1D array can be return using the for_loss method.
    It works for any type of ST_statistics. It can use a mask on the ST 
    coefficients, which is option-dependent
    
    Parameters
    ----------
    # Data type and Wavelet Transform
    - DT : str
        Type of data (1d, 2d planar, HealPix, 3d)
    - N0 : tuple
        initial size of array (can be multiple dimensions)
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
        
    # Data array parameters
    - Nb : int
        size of batch
    - Nc : int
        number of channel
        
    Attributes 
    ----------
    - parent parameters (DT,N0,J,L,WType,SC,jmin,jmax,dj,pbc,mask_MR,Nb,Nc)
    
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
    
    # ST statistics
    - S1, S2, S2p, S3, S4 : array of relevant size to store the ST statistics 
    
    '''

    ########################################
    def __init__(self,
                 DT, N0, J, L, WType,
                 SC, jmin, jmax, dj,
                 pbc, mask_MR,
                 Nb, Nc):
        '''
        Constructor, see details above.
        '''
        
        # Main parameters
        self.DT = DT
        self.N0 = N0
        
        # Wavelet transform related parameters
        self.J = J
        self.L = L
        self.WType = WType
        
        # Scattering transform related parameters
        self.SC = SC
        self.jmin = jmin
        self.jmax = jmax
        self.dj = dj
        self.pbc = pbc
        
        # Data related parameters
        self.Nb = Nb
        self.Nc = Nc
        
        # Additional transform/compression related parameters. While put to 
        # False/None for the initialization, their value are modified if these
        # methods are called by the scattering operator, or independently.
        self.norm = False
        self.S2_ref = None
        self.iso = False
        self.angular_ft = False
        self.scale_ft = False
        self.flatten = False
        self.mask_st = None

    ########################################
    def to_norm(self, norm=None, S2_ref=None):
        '''
        
        Normalize the ST statistics.
        
        Parameters
        ----------
        - norm : str
            type of norm ("S2", "S2_ref")
        - S2_ref : array
            array of reference S2 coefficients
            
        '''
        
        if self.iso:
            raise Exception(
                "Normalization can only be done before isotropization")  
        if self.angular_ft:
            raise Exception(
                "Normalization can only be done before angular ft")  
        if self.scale_ft:
            raise Exception(
                "Normalization can only be done before scate_ft")  
            
        if self.SC == "ScatCov":
            # perform normalization, to be done
            pass
        
        # Store normalization parameters
        self.norm = norm
        self.S2_Ref = S2_ref
        
        return self

    ########################################
    def to_iso(self):
        '''
        Isotropize the set of ST statistics
        
        Note: S2_ref is not isotropized since it is used before this step.
        
        EA: could probably be better vectorized, to be done.
        EA: to be done properly with the backend.
        EA: Sihao used .real for S3 and S4, to consider.
            
        '''
        
        if self.angular_ft:
            raise Exception(
                "Isotropization can only be done before angular ft")  
        if self.scale_ft:
            raise Exception(
                "Isotropization can only be done before scate_ft")  
            
        Nb, Nc = self.Nb, self.Nc
        J, L = self.J, self.L
            
        if self.SC == "ScatCov":
            
            # S1 and S2
            self.S1 = bk.mean(self.S1.mean, -1)       #(Nb,Nc,J,L) -> (Nb,Nc,J)
            self.S1 = bk.mean(self.S2.mean, -1)       #(Nb,Nc,J,L) -> (Nb,Nc,J)
            
            # S3 and S4
            S3iso = bk.zeros((Nb,Nc,J,J,L)) 
            S4iso = bk.zeros((Nb,Nc,J,J,J,L,L)) 
            for l1 in range(L):
                for l2 in range(L):
                    #(Nb,Nc,J,J,L,L) -> (Nb,Nc,J,J,L)
                    S3iso[...,(l2-l1)%L] += self.S3[...,l1,l2]
                    for l3 in range(L):
                        #(Nb,Nc,J,J,J,L,L,L) -> (Nb,Nc,J,J,J,L,L)
                        S4iso[...,(l2-l1)%L,(l3-l1)%L] += self.S4[...,l1,l2,l3]
                        
            self.S3 = S3iso / L
            self.S4 = S4iso / L

        # store isotropy parameter
        self.iso = True            
        
        return self
    
    ########################################
    def to_angular_ft(self):
        '''
        Angular harmonic transform on the ST statistcs 
        '''
        
        if self.scale_ft:
            raise Exception(
                "Angular_tf can only be done before scate_ft")  
            
         # perform angular transform, to be done
        if self.SC == "ScatCov":
            if self.iso:
                pass
            else:
                pass
            
        # store angular_ft parameter
        self.angular_ft = True
            
        return self
        
    ########################################
    def to_scale_ft(self):
        '''
        Angular scale transform on the ST statistcs 
        '''
            
         # perform scale transform, to be done
        if self.SC == "ScatCov":
            pass
            
        # store scale_ft parameter
        self.angular_ft = True
        
        return self
        
    ########################################
    def flatten(self, mask_st):
        '''
        Produce a 1d array that can be used for loss constructions.
        
        A mask can be used to select the coefficients from the initial 1d array.
        
        Parameters
        ----------
        - mask_st : binary 1d array
            mask for st coefficients after initial flattening
            
        Output 
        ----------
        - st_flatten : 1d array
            
        '''
        
        #return st_flatten
        
    ########################################
    def select(self, param):
        '''
        Select and give tensor in output 
        
        Parameters
        ----------
        - 
            
        Output 
        ----------
        - 
            
        '''
        
        output = 1

        return output 
        
    ########################################
    def plot_coeff(self, param):
        '''
        plot coeffs
        
        Parameters
        ----------
        - 
            
        Output 
        ----------
        - 
            
        '''
        
        #plot coeffs