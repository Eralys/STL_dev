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
                 Nb, Nc, wavelet_op):
        '''
        Constructor, see details above.
        '''
        
        # Main parameters
        self.DT = DT
        self.N0 = N0

        # Wavelet operator
        self.wavelet_op = wavelet_op
        
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
    def plot_coeff(self, b: int = 0, c: int = 0, new_figure: bool = True):
        """
        Reproduce the classical S1/S2/S3/S4 scattering plot for a given (batch, channel).

        Parameters
        ----------
        b : int
            Batch index (0 <= b < Nb).
        c : int
            Channel index (0 <= c < Nc).
        new_figure : bool
            If True, create a new figure. If False, plot into the existing one
            (useful to overlay multiple ST_statistic objects on the same panels).
        """

        # ---- extract S1..S4 for one (b,c) and convert to numpy ----
        def to_np(x):
            if isinstance(x, bk.Tensor):
                return x.detach().cpu().numpy()
            return np.asarray(x)

        if self.S1 is None:
            raise ValueError("S1 is None; nothing to plot.")

        S1_bc = to_np(self.S1[b, c])   # (J, L)
        S2_bc = to_np(self.S2[b, c])   # (J, L)
        S3_bc = to_np(self.S3[b, c])   # (J, J, L, L)
        S4_bc = to_np(self.S4[b, c])   # (J, J, J, L, L, L)

        # Put a fake 'image' dimension of size 1 to match the old plotting code
        S1 = S1_bc[None, ...]  # (1, J, L)
        S2 = S2_bc[None, ...]  # (1, J, L)

        # ---- build the compact index arrays for S3 and S4 as in your old code ----
        J = S1.shape[1]
        N_orient = S1.shape[2]

        # count combinations for S3 and S4
        n_s3 = 0
        n_s4 = 0
        for j3 in range(J):
            for j2 in range(j3 + 1):
                n_s3 += 1
                for j1 in range(j2 + 1):
                    n_s4 += 1

        j1_s3 = np.zeros(n_s3, dtype=int)
        j2_s3 = np.zeros(n_s3, dtype=int)

        j1_s4 = np.zeros(n_s4, dtype=int)
        j2_s4 = np.zeros(n_s4, dtype=int)
        j3_s4 = np.zeros(n_s4, dtype=int)

        n_s3 = 0
        n_s4 = 0
        for j3 in range(J):
            for j2 in range(0, j3 + 1):
                j1_s3[n_s3] = j2
                j2_s3[n_s3] = j3
                n_s3 += 1
                for j1 in range(0, j2 + 1):
                    j1_s4[n_s4] = j1
                    j2_s4[n_s4] = j2
                    j3_s4[n_s4] = j3
                    n_s4 += 1

        # Now we build compact S3 and S4 arrays with shape
        #   S3: (1, n_s3, L, L)
        #   S4: (1, n_s4, L, L, L)
        S3 = np.zeros((1, len(j1_s3), N_orient, N_orient), dtype=S3_bc.dtype)
        S4 = np.zeros((1, len(j1_s4), N_orient, N_orient, N_orient), dtype=S4_bc.dtype)

        for idx in range(len(j1_s3)):
            j1 = j1_s3[idx]
            j2 = j2_s3[idx]
            S3[0, idx, :, :] = S3_bc[j1, j2, :, :]

        for idx in range(len(j1_s4)):
            j1 = j1_s4[idx]
            j2 = j2_s4[idx]
            j3 = j3_s4[idx]
            S4[0, idx, :, :, :] = S4_bc[j1, j2, j3, :, :, :]

        # ---- now we reproduce your original plot_scat(S1,S2,S3,S4) ----
        color = ['b', 'r', 'orange', 'pink']
        symbol = ['', ':', '-', '.']

        if new_figure:
            plt.figure(figsize=(16, 12))

        # ----- S1 -----
        plt.subplot(2, 2, 1)
        for k in range(min(4, N_orient)):
            plt.plot(S1[0, :, k],
                     color=color[k % len(color)],
                     label=rf'$\Theta = {k}$')
        plt.legend(frameon=False, ncol=2)
        plt.xlabel(r'$J_1$')
        plt.ylabel(r'$S_1$')
        plt.yscale('log')

        # ----- S2 -----
        plt.subplot(2, 2, 2)
        for k in range(min(4, N_orient)):
            plt.plot(S2[0, :, k],
                     color=color[k % len(color)],
                     label=rf'$\Theta = {k}$')
        plt.xlabel(r'$J_1$')
        plt.ylabel(r'$S_2$')
        plt.yscale('log')

        # ----- S3 -----
        plt.subplot(2, 2, 3)
        # nidx to separate groups of constant j1
        nidx = np.concatenate(
            [np.zeros([1], dtype=int),
             np.cumsum(np.bincount(j1_s3, minlength=J))],
            axis=0
        )
        l_pos = []
        l_name = []
        for i in np.unique(j1_s3):
            idx = np.where(j1_s3 == i)[0]
            for k in range(min(4, N_orient)):
                for l in range(min(4, N_orient)):
                    if i == 0:
                        plt.plot(j2_s3[idx] + nidx[i],
                                 S3[0, idx, k, l],
                                 symbol[l % len(symbol)],
                                 color=color[k % len(color)],
                                 label=rf'$\Theta = {k},{l}$')
                    else:
                        plt.plot(j2_s3[idx] + nidx[i],
                                 S3[0, idx, k, l],
                                 symbol[l % len(symbol)],
                                 color=color[k % len(color)])
            l_pos += list(j2_s3[idx] + nidx[i])
            l_name += [f"{j1_s3[m]},{j2_s3[m]}" for m in idx]

        plt.legend(frameon=False, ncol=2)
        plt.xticks(l_pos, l_name, fontsize=6)
        plt.xlabel(r"$j_{1},j_{2}$", fontsize=9)
        plt.ylabel(r"$S_{3}$", fontsize=9)

        # ----- S4 -----
        plt.subplot(2, 2, 4)
        nidx = 0
        l_pos = []
        l_name = []
        for i in np.unique(j1_s4):
            for j in np.unique(j2_s4):
                idx = np.where((j1_s4 == i) & (j2_s4 == j))[0]
                for k in range(min(4, N_orient)):
                    for l in range(min(4, N_orient)):
                        for m in range(min(4, N_orient)):
                            if i == 0 and j == 0 and m == 0:
                                plt.plot(j2_s4[idx] + j3_s4[idx] + nidx,
                                         S4[0, idx, k, l, m],
                                         symbol[l % len(symbol)],
                                         color=color[k % len(color)],
                                         label=rf'$\Theta = {k},{l},{m}$')
                            else:
                                plt.plot(j2_s4[idx] + j3_s4[idx] + nidx,
                                         S4[0, idx, k, l, m],
                                         symbol[l % len(symbol)],
                                         color=color[k % len(color)])
                l_pos += list(j2_s4[idx] + j3_s4[idx] + nidx)
                l_name += [f"{j1_s4[m]},{j2_s4[m]},{j3_s4[m]}" for m in idx]
            # increment nidx to separate groups of constant j1
            sel = (j1_s4 == i)
            if np.any(sel):
                span = (j2_s4[sel] + j3_s4[sel])
                nidx += int(np.max(span) - np.min(span) + 1)

        plt.legend(frameon=False, ncol=2)
        plt.xticks(l_pos, l_name, fontsize=6, rotation=90)
        plt.xlabel(r"$j_{1},j_{2},j_{3}$", fontsize=9)
        plt.ylabel(r"$S_{4}$", fontsize=9)

        plt.tight_layout()
        plt.show()
