# This program is for reproduging CHien and Kim (2023)
import numpy as np

def _Covaiance(arr1, arr2, axes=(-1, -2)):
    
    fft1 = np.fft.fft(arr1, axis=axes[0]);
    fft1 = np.fft.ifft(fft1, axis=axes[1]) * arr1.shape[axes[1]];
    
    fft2 = np.fft.fft(arr2, axis=axes[0]);
    fft2 = np.fft.ifft(fft2, axis=axes[1]) * arr2.shape[axes[1]];

    cs = (fft1 * fft2.conj()) / np.prod(arr1.shape[axes]);
    
    cs_smooth = np.empty(cs.shape, dtype=complex);
    
    kernel = np.array([1, 2, 1]) / 4.0;
    
    for i in range(cs.shape[0]):
        cs_smooth[i] = np.convolve(cs[i], kernel, mode='same')
    
    for i in range(cs.shape[1]):
        cs_smooth[:, i] = np.convolve(cs_smooth[:, i], kernel, mode='same')
    
    return cs_smooth

def growth_rate(arr1, arr2, axes=(-1, -2)):
    
    var = _Covaiance(arr2, arr2, axes).mean(axis=0);
    cov = _Covaiance(arr1, arr2, axes).mean(axis=0);
    
    sigma = 2 * cov.real / var;
    return sigma
    
def Coherence(arr1, arr2, axes=(-1, -2)):
    var1 = _Covaiance(arr1, arr1, axes).mean(axis=0);
    var2 = _Covaiance(arr2, arr2, axes).mean(axis=0);
    cov  = _Covaiance(arr1, arr2, axes).mean(axis=0);
    
    return (cov * cov.conj()).real / (var1 * var2);
