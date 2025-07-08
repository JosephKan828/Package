import numpy as np


class EquatorialWaveDispersion:
    """
    Class to compute dispersion curves of equatorial shallow water waves
    following the Matsuno (1966) formulation.

    Supports:
        - Mixed Rossby-Gravity (MRG) wave
        - Inertial Gravity (IG) waves (n=0,1,2)
        - Equatorial Rossby (ER) wave (n=1)
        - Kelvin wave

    Parameters:
        nWaveType (int): Number of wave types to compute (max 6)
            0: MRG (antisymmetric)
            1: IG (n=0, antisymmetric)
            2: IG (n=2, antisymmetric)
            3: ER (n=1, symmetric)
            4: Kelvin (symmetric)
            5: IG (n=1, symmetric)
        nPlanetaryWave (int): Number of zonal wavenumber samples to compute
        rlat (float): Latitude in radians (usually 0.0 for equator)
        Ahe (list of float): Equivalent depths [m] for vertical modes

    Methods:
        compute(): Computes the dispersion frequencies and wavenumbers
        mrg_wave(k, c): MRG wave dispersion
        ig_wave_n0(k, c): IG wave with n=0
        ig_wave_n(k, c, he, n): IG wave with arbitrary meridional mode n
        er_wave_n(k, c, n): Equatorial Rossby wave with mode n
        kelvin_wave(k, c): Kelvin wave dispersion

    Returns:
        Afreq (np.ndarray): Frequency [1/day], shape (nWaveType, nEquivDepth, nPlanetaryWave)
        Apzwn (np.ndarray): Zonal wavenumber (s), shape (nWaveType, nEquivDepth, nPlanetaryWave)
    """

    def __init__(self, nWaveType=6, nPlanetaryWave=50, rlat=0.0, Ahe=[50, 25, 12]):
        # Constants
        self.pi = np.pi
        self.radius = 6.37122e6
        self.g = 9.80665
        self.omega = 7.292e-5
        self.fillval = np.nan

        # Inputs
        self.nWaveType = nWaveType
        self.nPlanetaryWave = nPlanetaryWave
        self.Ahe = Ahe
        self.rlat = rlat

        # Derived quantities
        self.Beta = 2.0 * self.omega * np.cos(np.abs(self.rlat)) / self.radius
        self.ll = 2.0 * self.pi * self.radius * np.cos(np.abs(self.rlat))

        # Output arrays
        self.Afreq = np.empty((nWaveType, len(Ahe), nPlanetaryWave))
        self.Apzwn = np.empty((nWaveType, len(Ahe), nPlanetaryWave))

    def mrg_wave(self, k, c):
        """Dispersion relation for mixed Rossby-gravity (MRG) wave"""
        if k < 0:
            dell = np.sqrt(1.0 + (4.0 * self.Beta) / (k**2 * c))
            return k * c * (0.5 - 0.5 * dell)
        elif k == 0:
            return np.sqrt(c * self.Beta)
        return self.fillval

    def ig_wave_n0(self, k, c):
        """Dispersion relation for n=0 inertial gravity wave"""
        if k == 0:
            return np.sqrt(c * self.Beta)
        elif k > 0:
            dell = np.sqrt(1.0 + (4.0 * self.Beta) / (k**2 * c))
            return k * c * (0.5 + 0.5 * dell)
        return self.fillval

    def ig_wave_n(self, k, c, he, n):
        """Dispersion relation for IG wave with meridional mode n"""
        dell = self.Beta * c
        freq = np.sqrt((2 * n + 1) * dell + self.g * he * k**2)
        for _ in range(5):
            freq = np.sqrt((2 * n + 1) * dell + self.g * he * k **
                           2 + self.g * he * self.Beta * k / freq)
        return freq

    def kelvin_wave(self, k, c):
        """Dispersion relation for Kelvin wave"""
        return k * c

    def er_wave_n(self, k, c, n):
        """Dispersion relation for equatorial Rossby wave with mode n"""
        if k < 0:
            dell = (self.Beta / c) * (2 * n + 1)
            return -self.Beta * k / (k**2 + dell)
        return self.fillval

    def compute(self):
        """
        Compute dispersion curves for all wave types and equivalent depths.

        Returns:
            Afreq (np.ndarray): Frequency [1/day],
                                shape (nWaveType, nEquivDepth, nPlanetaryWave)
            Apzwn (np.ndarray): Zonal wavenumber (s),
                                shape (nWaveType, nEquivDepth, nPlanetaryWave)
        """
        for wave_id in range(self.nWaveType):
            for ed, he in enumerate(self.Ahe):
                c = np.sqrt(self.g * he)
                # L = np.sqrt(c / self.Beta)

                for wn in range(self.nPlanetaryWave):
                    s = np.linspace(-20, 20, self.nPlanetaryWave)[wn]
                    k = 2.0 * self.pi * s / self.ll
                    freq = self.fillval

                    if wave_id == 0:
                        freq = self.mrg_wave(k, c)
                    elif wave_id == 1:
                        freq = self.ig_wave_n0(k, c)
                    elif wave_id == 2:
                        freq = self.ig_wave_n(k, c, he, n=2)
                    elif wave_id == 3:
                        freq = self.er_wave_n(k, c, n=1)
                    elif wave_id == 4:
                        freq = self.kelvin_wave(k, c)
                    elif wave_id == 5:
                        freq = self.ig_wave_n(k, c, he, n=1)

                    self.Apzwn[wave_id, ed, wn] = s
                    self.Afreq[wave_id, ed, wn] = (
                        freq / (2 * self.pi) *
                        86400 if freq != self.fillval else self.fillval
                    )

        return self.Afreq, self.Apzwn
