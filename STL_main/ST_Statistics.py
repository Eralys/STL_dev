# -*- coding: utf-8 -*-
"""
Main structure of STL

Tentative proposal by EA 
"""
                    
###############################################################################
###############################################################################

class ST_Statistics:
    '''
    Class whose instances correspond to an set of scattering statistics
    The set of statistics is built by the ST_operator method, which use the 
    __init__ method.
    This class contains methods that allow to deal with ST statistics in an 
    unified manner. Parts of these methods can be applied directly through the 
    ST_operator implementation.
    This class is DT-independent.
    
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
    - jmin : int
        minimum scale at which the ST statistics are computed
    - jmax : int
        maximum scale at which the ST statistics are computed
    - dj : int
        maximum scale ratio at which the ST statistics are computed
    
    Attributes 
    ----------
    - parent parameters (DT,N,J,L,WTWTypeSC,jmin,jmax,dj)
    - norm : str
        current normalisation of the ST statistics
    - iso : bool
        isotropic status of the ST statistics
    - to do : something about j/l transform and thresholding
    - S1, S2, S2p, S3, S4 : array of relevant size to store the ST statistics 
    
    Questions and to do
    ----------
    '''

    ########################################
    def __init__(self, N, J, L, WType, SC, jmin, jmax, dj):
        '''
        Constructor, see details above.
        '''
        # Parameters of the constructor
        self.N = N
        self.J = J
        self.L = L
        self.WType = WType
        self.SC = SC
        
        # Parameter of the computation
        self.jmin = jmin
        self.jmax = jmax
        self.dj = dj
        
        # Initialization of additional transform parameters
        self.norm = None
        self.iso = False
        
    ########################################
    def norm(self, param):
        '''
        Normalize
        
        Parameters
        ----------
        - 
            
        Output 
        ----------
        - 
            
        '''
        
        #modify self


    ########################################
    def to_iso(self, param):
        '''
        Isotropize
        
        Parameters
        ----------
        - 
            
        Output 
        ----------
        - 
            
        '''
        
        #modify self

    
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
    def ang_ft(self, param):
        '''
        angular harmonic transform
        
        Parameters
        ----------
        - 
            
        Output 
        ----------
        - 
            
        '''
        
        #modify self
        
    ########################################
    def scale_ft(self, param):
        '''
        scale harmonic transform
        
        Parameters
        ----------
        - 
            
        Output 
        ----------
        - 
            
        '''
        
        #modify self
        
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