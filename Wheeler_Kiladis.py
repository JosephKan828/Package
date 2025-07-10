import numpy as np
from scipy.signal import detrend as scipy_detrend
from scipy.ndimage import convolve1d


class SpaceTimeSpectrumAnalyzer:
    def __init__(
            self,
            dims: dict[str, np.ndarray],
            exp_list: list[str],
            lat_range: tuple[float, float] = (-5, 5)
    ):
        """
        Initialize the analyzer with dimension info, experiment names, and latitude limits.

        Parameters:
        - dims: Dictionary of dimension arrays (e.g., lat, lon, lev)
        - exp_list: List of experiment names
        - lat_range: Tuple of latitude bounds for subsetting (default: (-5, 5))
        """
        self.lat_range = lat_range
        self.dims = dims
        self.lat_lim = np.where((self.dims["lat"] >= lat_range[0]) & (
            self.dims["lat"] <= lat_range[1]))[0]
        self.dims["lat"] = self.dims["lat"][self.lat_lim]
        if "lev" in self.dims:
            self.converter = (
                1000.0 / self.dims["lev"][None, :, None, None]) ** (-0.285)
        else:
            self.converter = None

    @staticmethod
    def detrend(data: np.ndarray, axis: int) -> np.ndarray:
        """
        Linearly detrend the data along the specified axis.

        Parameters:
        - data: Input array
        - axis: Axis along which to detrend

        Returns:
        - Detrended array
        """
        return scipy_detrend(data, axis=axis, type='linear')

    def chunk_and_window(self, data: np.ndarray) -> np.ndarray:
        """
        Split data into overlapping 120-point chunks with a 60-point stride,
        and apply a Hanning window.

        Parameters:
        - data: Input data array (3D or 4D)

        Returns:
        - Windowed chunks as a NumPy array
        """
        shape = data.shape
        if len(shape) == 4:
            hanning = np.hanning(120)[:, None, None, None]
        elif len(shape) == 3:
            hanning = np.hanning(120)[:, None, None]
        else:
            raise ValueError("Unsupported data shape. Expect 3D or 4D array.")

        return np.array([
            self.detrend(data[i*60:i*60+120], 0) * hanning
            for i in range(360//120+2)
        ])

    def fft_freqs(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the wavenumber and frequency axes.

        Returns:
        - Tuple of (wavenumbers, frequencies)
        """
        wn = np.fft.fftshift(np.fft.fftfreq(
            self.dims["lon"].size, d=1/self.dims["lon"].size)).astype(int)
        fr = np.fft.fftshift(np.fft.fftfreq(120, d=1/4))
        return wn, fr

    @staticmethod
    def compute_background(data: np.ndarray) -> np.ndarray:
        """
        Smooth the power spectrum to estimate background using repeated 1D convolution.

        Parameters:
        - data: Input 2D power spectrum (freq x wavenumber)

        Returns:
        - Smoothed background power spectrum
        """
        data = data.copy()
        kernel = np.array([1, 2, 1]) / 4.0
        half_freq = data.shape[0] // 2

        for _ in range(10):
            data = convolve1d(data, kernel, axis=0, mode="reflect")

        for _ in range(10):
            data[:half_freq] = convolve1d(
                data[:half_freq], kernel, axis=1, mode="reflect")

        for _ in range(40):
            data[half_freq:] = convolve1d(
                data[half_freq:], kernel, axis=1, mode="reflect")

        return data

    @staticmethod
    def normalize_power(power: np.ndarray, background: np.ndarray) -> np.ndarray:
        """
        Normalize power spectrum by background.

        Parameters:
        - power: Raw power spectrum
        - background: Background spectrum

        Returns:
        - Normalized spectrum
        """
        return power / background

    @staticmethod
    def symm_asymm(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute symmetric and antisymmetric components relative to the equator.

        Parameters:
        - data: Input data array

        Returns:
        - Tuple of (symmetric, antisymmetric) components
        """
        symm = (data + np.flip(data, axis=2)) / 2.0
        asym = (data - np.flip(data, axis=2)) / 2.0
        return symm, asym

    def process_spectrum(self, anomalies: np.ndarray) -> np.ndarray:
        """
        Process anomalies to compute the normalized power spectrum.

        Steps:
        - Split into symmetric and antisymmetric components
        - Apply chunking and windowing
        - Compute 2D FFT and average power
        - Filter to positive frequencies
        - Normalize by background spectrum

        Parameters:
        - anomalies: Input anomaly field (time, lat, lon) or (time, lev, lat, lon)

        Returns:
        - Normalized power spectrum (positive frequencies only)
        """
        symm_data, asym_data = self.symm_asymm(anomalies)

        symm_chunks = self.chunk_and_window(symm_data)
        asym_chunks = self.chunk_and_window(asym_data)

        def power_spectrum(chunks):
            fft_data = np.fft.fft2(chunks, axes=(1, -1))
            ps = np.abs(fft_data) ** 2
            return ps.mean(axis=0)

        symm_power = power_spectrum(symm_chunks)
        asym_power = power_spectrum(asym_chunks)

        total_power = (symm_power + asym_power) / 2.0
        fr = self.fft_freqs()[1]
        total_power = total_power[fr > 0]

        background = self.compute_background(total_power)
        normalized = self.normalize_power(total_power, background)

        return normalized
