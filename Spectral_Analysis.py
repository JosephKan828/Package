import numpy as np;

def PowerSpectrum_1D( data: np.ndarray, axis=-1 ):

    data: np.ndarray = np.array( data );

    if len( data.shape ) != 1:
        raise DimensionError( "The array must be 1-dimensional array." );

    N: int = data.size; # size of 1d data

    fr: np.ndarray = np.fft.fftfreq( N, d=1/N );
    print( fr )
    data_fft  : np.ndarray = np.fft.fft( data, axis=axis ); # apply FFT on data

    power_spec: np.ndarray = ( data_fft * data_fft.conj() ) / ( float( len( data ) )**2 ); # Compute power spectrum

    power_spec = power_spec[fr>0] * 2.0    

    return power_spec;

def PowerSpectrum_2D( data, axes=(0, -1) ):
    data_fft = np.fft.rfft( data, axis=axes[0] ) * 2.0;
    data_fft = np.fft.ifft( data_fft, axis=axes[1] ) * data.shape[axes[1]];

    power_spec = ( data_fft * data_fft.conj() ) / ( data.shape[axes[0]] * data.shape[axes[1]] ) ** 2.0 / 2;

    return power_spec.real[1:]

