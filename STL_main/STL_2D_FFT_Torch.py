"""
Created on Wed Nov 14:07 2018
"""
import numpy as np
import torch

class STL_2D_FFT_Torch:
    """
    Class for 2D planar STL FFT using PyTorch
    """

    @staticmethod
    def to_array(data):
        """
        Convert input to a PyTorch array.
        """
        array = torch.as_tensor(data)
        return array

    @staticmethod    
    def cov(array1, fourier_status1, array2, fourier_status2, mask, remove_mean=False):
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
        """
        self.array = self.__class__.to_array(data)
        self.fourier_status = fourier_status

    def findN(self):
        """
        Find the dimensions of the 2D planar data, which are expected to be the 
        last two dimensions of the array.

        Returns
        -------
        N : tuple of int
            The spatial dimensions  of the 2D planar data.
        """
        
        # Get the shape of the tensor
        if not self.fourier_status:
            shape = self.array.shape
        # Return the last two dimensions
        return (shape[-2], shape[-1])

    def copy(self):
        """
        Copy of the array attribute.
        
        Returns
        -------
        torch.Tensor
            A copy of the input tensor.
        """
        
        return self.array.clone()
    

    
    def modulus(self):
        """
        Take the modulus of the array attribute.

        Parameters
        ----------
        array : torch.Tensor
            Input tensor.
        
        Returns
        -------
        torch.Tensor
            Modulus of input tensor.
        """
    
        return self.array.abs()

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
            if square is False:
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

    def subsampling(self, N0, dg, dg_out, mask_MR):
        """
        Downsample the data to the specified resolution.
        
        Note: Masks are not supported in this data type.
        
        Parameters
        ----------
        N0 : tuple of int
            Initial resolution of the data.
        dg : int
            Current downsampling factor of the data.
        dg_out : int
            Desired downsampling factor of the data.
        mask_MR : None
            Placeholder for mask, not used in this function.
        
        Returns
        -------
        torch.Tensor
            Downsampled data at the desired downgrading factor dg_out.
        fourier : bool
            Indicates whether output array is in Fourier space.        
        """

        N0
        
        if mask_MR is not None:
            raise Exception("Masks are not supported in DT1") 
            
        if dg_out == dg:
            return array, Fourier
        
        # Tuning parameter to keep the aspect ratio and a unified resolution
        min_x, min_y = 8, 8
        if N0[0] > N0[1]:
            min_x = int(min_x * N0[0]/N0[1])
        elif N0[1] > N0[0]:
            min_y = int(min_y * N0[1]/N0[0])

        # Identify the new dimensions
        dx = int(max(min_x, N0[0] // 2**(dg_out + 1)))
        dy = int(max(min_y, N0[1] // 2**(dg_out + 1)))
        
        # Check expected current dimensions
        dx_cur = int(max(min_x, N0[0] // 2**(dg + 1)))
        dy_cur = int(max(min_y, N0[1] // 2**(dg + 1)))
        
        # Perform downsampling if necessary
        if dx != dx_cur or dy != dy_cur:
            
            # Fourier transform if in real space
            if not Fourier:
                array = torch.fft.fft2(array, norm="ortho")
                Fourier = True
            
            # Downsampling in Fourier
            array_dg = torch.cat(
                (torch.cat(
                    (array[...,:dx, :dy], array[...,-dx:, :dy]), -2),
                torch.cat(
                    (array[...,:dx, -dy:], array[...,-dx:, -dy:]), -2)
                ),-1) * np.sqrt(dx * dy / dx_cur / dy_cur)
            return array_dg, Fourier
            
        else:
            return array, Fourier

if __name__ == "__main__":
    # Simple test
    data = np.random.rand(4, 5, 6)
    stl_fft = STL_2D_FFT_Torch(data)
    print("Array:\n", stl_fft.array)
    print("Modulus:\n", stl_fft.modulus())
    print("Mean:\n", stl_fft.mean())


    