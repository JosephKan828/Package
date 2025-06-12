#This module contains functions for spectral analysis of time series data
import numpy as np;

class Fourier:
    def __init__(
        self,
        arr: np.ndarray,
        axes: tuple[int, int] = (-2, -1),
        ):
        
        self.arr   : np.ndarray      = arr;
        self.axes  : tuple[int, int] = axes;
        self.shape : tuple[int]      = arr.shape;
    
    def fft2d(self):
        
        fft: np.ndarray = np.fft.fft(self.arr, axis=self.axes[0]);
        fft: np.ndarray = np.fft.ifft(fft, axis=self.axes[1]) * self.shape[self.axes[1]];
        
        return fft;
    
    def ifft2d(self):
        
        ifft: np.ndarray = np.fft.ifft(self.arr, axis=self.axes[0]);
        ifft: np.ndarray = np.fft.fft(ifft, axis=self.axes[1]) / self.shape[self.axes[1]];
        
        return ifft;
    
class SpectralAnalysis(Fourier):
    def __init__(
        self,
        arr: np.ndarray,
        axes: tuple[int, int] = (-2, -1)
    ):
        super().__init__(arr, axes);
        self.fft : np.ndarray = self.fft2d();
        self.ifft: np.ndarray = self.ifft2d();
    
    def power_spectrum(
        self
        ) -> np.ndarray:
        ps: np.ndarray = (self.fft * self.ifft.conj()).real / (self.shape[self.axes[0]] * self.shape[self.axes[1]]) * 2;
        
        return ps;
      
