# -*- coding: utf-8 -*-
"""
Main structure of STL

Tentative proposal by EA 
"""

import torch as bk # mean, zeros
import matplotlib.pyplot as plt
import numpy as np

                    
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
                 Nb, Nc,wavelet_op):
        '''
        Constructor, see details above.
        '''
        
        # Main parameters
        self.DT = DT
        self.N0 = N0
        
        # Wavelet transform related parameters
        self.wavelet_op=wavelet_op
        self.J = self.wavelet_op.J
        self.L = self.wavelet_op.L
        self.WType = self.wavelet_op.WType
        
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
            type of norm (“S2”, “S2_ref”)
        - S2_ref : array
            array of reference S2 coefficients
        '''
        
        # Check the proper ordering
        if self.iso:
            raise Exception("Normalization can only be done before isotropization")
        if self.angular_ft:
            raise Exception("Normalization can only be done before angular ft")
        if self.scale_ft:
            raise Exception("Normalization can only be done before scate_ft")
            
        # Leave the function if no normalization is required
        if norm is None:
            pass
            
        # Store_ref normalization
        elif norm == "store_ref":
            # Verifications
            if self.norm:
                raise Exception("ST statistics are already normalized")
            # Perform normalization and store reference
            if self.SC == "ScatCov":
                S2_ref = self.S2*1.0
                self.S1 = self.S1 / bk.sqrt(S2_ref)
                self.S2 = self.S2 / S2_ref
                self.S3 = self.S3 / bk.sqrt(S2_ref[:,:,:,None,:,None] * S2_ref[:,:,None,:,None,:])
                self.S4 = self.S4 / bk.sqrt(S2_ref[:,:,:,None,None,:,None,None] * S2_ref[:,:,None,:,None,None,:,None])
            self.norm = True
            self.S2_ref = S2_ref

        # Load_ref normalization
        elif norm == "load_ref":
            # Verifications
            if self.norm:
                raise Exception("ST statistics are already normalized")
            if S2_ref is None:
                raise Exception("S2_ref should be given")
            # Perform normalization and store reference
            if self.SC == "ScatCov":
                self.S1 = self.S1 / bk.sqrt(S2_ref)
                self.S2 = self.S2 / S2_ref
                self.S3 = self.S3 / bk.sqrt(S2_ref[:,:,:,None,:,None] * S2_ref[:,:,None,:,None,:])
                self.S4 = self.S4 / bk.sqrt(S2_ref[:,:,:,None,None,:,None,None] * S2_ref[:,:,None,:,None,None,:,None])
            # Store normalization parameters
            self.norm = True
            self.S2_ref = S2_ref
            
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
    def to_flatten(self, mask_st=None):
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
        
        # Collect all S1,S2,S3,S4 into a list
        stats = [self.S1, self.S2, self.S3, self.S4]

        # Flatten each, remove NaNs, concat
        flattened_list = []
        for S in stats:
            # S may contain NaNs → keep only non-NaNs
            S_flat = S.reshape(-1)
            valid = ~bk.isnan(S_flat)
            flattened_list.append(S_flat[valid])

        # Concatenate all statistics into a single 1D vector
        st_flatten = bk.cat(flattened_list, dim=0)

        # Optional mask after nan-removal
        if mask_st is not None:
            mask_st = bk.as_tensor(mask_st, dtype=bk.bool, device=st_flatten.device)
            if mask_st.numel() != st_flatten.numel():
                raise ValueError(
                    f"mask_st length {mask_st.numel()} does not match "
                    f"flattened statistic length {st_flatten.numel()}."
                )
            st_flatten = st_flatten[mask_st]

        return st_flatten
        
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
    def _to_np(self,x):
        if isinstance(x, bk.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)


    ########################################
    def plot_coeff(self, b=0, c=0, hold=False, color=None, figsize=(16, 12)):
        """
        Plot scattering covariance coefficients as curves.

        Parameters
        ----------
        b : int
            Batch index.
        c : int
            Channel index.
        hold : bool
            If False, clear the figure before plotting new curves.
        color : str or None
            Color for curves (matplotlib format).
        figsize : tuple
            Size of figure when hold=False.

        Behavior
        --------
        - S1(j,l) : one curve per orientation l.
        - S2(j,l) : one curve per orientation l.
        - S3(j1,j2,l1,l2) :
             * Panel per (l1,l2)
             * For each panel: x=j1, curves for each j2.
        - S4(j1,j2,j3,l1,l2,l3) :
             * Panel per (l1,l2,l3)
             * For each panel: x=j1, curves for each j2 and j3 fixed.
        """

        if not hold:
            plt.figure(figsize=figsize)
            plt.clf()

        Nb, Nc, J, L = self.Nb, self.Nc, self.J, self.L
        if b >= Nb or c >= Nc:
            raise IndexError("b or c index out of range.")

        ###########################################################################
        # -----------  S1 : (J, L)  ----------- #
        ###########################################################################
        if hasattr(self, "S1") and self.S1 is not None:
            S1 = self._to_np(self.S1[b, c])  # (J, L)

            plt.subplot(2, 2, 1)
            for l in range(L):
                y = S1[:, l]
                mask = ~np.isnan(y)
                if np.any(mask):
                    plt.plot(np.arange(J)[mask], y[mask], color=color, label=f"L={l}")
            plt.title("S1(j1,l)")
            plt.xlabel("j1")
            plt.ylabel("S1")
            plt.grid(True)
            plt.legend()

        ###########################################################################
        # -----------  S2 : (J, L)  ----------- #
        ###########################################################################
        if hasattr(self, "S2") and self.S2 is not None:
            S2 = self._to_np(self.S2[b, c])  # (J, L)

            plt.subplot(2, 2, 2)
            for l in range(L):
                y = S2[:, l]
                mask = ~np.isnan(y)
                if np.any(mask):
                    plt.plot(np.arange(J)[mask], y[mask], color=color, label=f"L={l}")
            plt.title("S2(j1,l)")
            plt.xlabel("j1")
            plt.ylabel("S2")
            plt.grid(True)
            plt.legend()

        ###########################################################################
        # -----------  S3 : (J,J,L,L)  ----------- #
        ###########################################################################
        if hasattr(self, "S3") and self.S3 is not None:
            S3 = self._to_np(self.S3[b, c])  # (J, J, L, L)

            # On ouvre une figure dédiée pour S3 si hold=False (car beaucoup de panels)
            if not hold:
                fig3 = plt.figure(figsize=(14, 10))
                fig3.suptitle("S3(j1,j2,l1,l2)")

            panel = 1
            for l1 in range(L):
                for l2 in range(L):
                    if not hold:
                        plt.subplot(L, L, panel)
                    panel += 1

                    # S3[j1, j2] (fix l1,l2)
                    S = S3[:, :, l1, l2]  # shape (J,J)

                    for j2 in range(J):
                        y = S[:, j2]
                        mask = ~np.isnan(y)
                        if np.any(mask):
                            plt.plot(np.arange(J)[mask], y[mask],
                                     color=color, label=f"j2={j2}")

                    if not hold:
                        plt.title(f"S3 l1={l1}, l2={l2}")
                        plt.xlabel("j1")
                        plt.ylabel("coef")
                        plt.grid(True)
                        # (pas de légende pour éviter surcharge visuelle)

        ###########################################################################
        # -----------  S4 : (J,J,J,L,L,L)  ----------- #
        ###########################################################################
        if hasattr(self, "S4") and self.S4 is not None:
            S4 = self._to_np(self.S4[b, c])  # (J, J, J, L, L, L)

            # Figure dédiée
            if not hold:
                fig4 = plt.figure(figsize=(16, 12))
                fig4.suptitle("S4(j1,j2,j3,l1,l2,l3)")

            panel = 1
            for l1 in range(L):
                for l2 in range(L):
                    for l3 in range(L):
                        # trop de sous-graphiques si L grand : à ton goût !
                        if not hold:
                            plt.subplot(L**2, L, panel)
                        panel += 1

                        # S4[j1,j2,j3,l1,l2,l3]
                        S = S4[:, :, :, l1, l2, l3]  # (J,J,J)

                        for j2 in range(J):
                            for j3 in range(J):
                                y = S[:, j2, j3]
                                mask = ~np.isnan(y)
                                if np.any(mask):
                                    plt.plot(np.arange(J)[mask], y[mask],
                                             color=color)

                        if not hold:
                            plt.title(f"S4 l1={l1},l2={l2},l3={l3}")
                            plt.xlabel("j1")
                            plt.ylabel("coef")
                            plt.grid(True)

        if not hold:
            plt.tight_layout()
            plt.show()
