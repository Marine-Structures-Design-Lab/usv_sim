import numpy as np
from pathlib import Path

script_dir = Path(__file__).resolve().parent
ocean_path = script_dir.parent / "Ocean" / "NOAA_NorthAtlantic.nc"


class RAO:
    """
    Creates an RAO and generates a response spectra for the RAO
    """

    def __init__(self, speed, heading, freq, amplitude, spectrum, phase_shift=None):
        """
        RAO Constructor
        --------------------------------------------------------------------------
        Parameters
        -------------------------------------------------------------------------
        :param speed: float, speed value for the RAO(knots)
        :param heading: float, heading value fpr the RAO(degrees)
        :param freq: 1D array of frequencies
        :param amplitude: 1D array of amplitude factors related to the frequencies
        :param spectrum: 1D array of floats, wave spectrum
        :param phase_shift(optional): 1D array of phase shifts related to the frequencies
        """
        self.speed = speed
        self.heading = heading
        self.freq = freq
        self.amplitude = amplitude
        self.spectrum = spectrum
        self.phase_shift = phase_shift

    def generate_response_spectra(self, amplitude):
        """
        Generates the response spectra for a given wave spectrum and RAO.
        -----------------------------------------------------------------
        Parameters
        -----------------------------------------------------------------
        amplitude: list of floats
        ----------------------------------------------------------------
        Returns:
            list of floats: response spectra for each frequency
        """
        response_spectra = np.zeros(len(self.freq))
        for i in range(len(self.freq)):
            response_spectra[i] = amplitude[i] ** 2 * self.spectrum[i]
        return response_spectra

    def JensenHeavePitchRAO(self, L, B0, Cb, T):
        """
        Parameters
        ----------
        L : Float
            Ovreall Length, meters.
        B0 : Float
            Breadth at waterline, meters.
        Cb : Float
            Block coefficient.
        T : Float
            Ovreall draft, meters.
        Returns
        -------
        Tuple of four numpy arrays, first is encounter frequencies, second is
        the ground-based frequency, and third is the heave RAO values, fourth
        is the pitch RAO values

        """
        # Constants
        g = 9.81

        # Effective b
        B = B0 * Cb

        # Convert speed to m/s
        Vms = self.speed * 0.514444

        # Convert heading to radians
        BetaRad = np.radians(self.heading)

        # Calculate Wave Number
        K = self.freq**2 / g

        # Alpha is related to the speed/heading combo
        FroudeNumber = Vms / np.sqrt(g * L)
        Alpha = 1.0 - FroudeNumber * (K * L) ** (0.5) * np.cos(BetaRad)

        # Calculate omega bar:
        OmegaBar = self.freq * Alpha

        A = (
            2.0
            * np.sin(np.square(OmegaBar) * B / (2.0 * g))
            * np.exp(-np.square(OmegaBar) * T / g)
        )

        # Forcing functions
        k_eff = np.abs(K * np.cos(BetaRad))
        Kappa = np.exp(-k_eff * T)

        script_f = np.power(
            np.square(1.0 - K * T)
            + np.square(np.square(A) / (K * B * np.power(Alpha, 3.0))),
            0.5,
        )

        F = Kappa * script_f * (2.0 / (k_eff * L)) * np.sin(k_eff * L / 2.0)

        G = (
            Kappa
            * script_f
            * 24.0
            / (np.square(k_eff * L) * L)
            * (np.sin(k_eff * L / 2.0) - k_eff * L / 2.0 * np.cos(k_eff * L / 2.0))
        )

        # print(A)
        # print(Alpha)

        eta = 1.0 / (
            np.power(
                np.square(1.0 - 2.0 * K * T * np.square(Alpha))
                + np.square(np.square(A) / (K * B * np.square(Alpha))),
                0.5,
            )
        )

        HeaveRAO = np.abs(eta * F)
        PitchRAO = np.abs(eta * G)

        return (HeaveRAO, PitchRAO)


class RAO_Library:
    """
    Stores RAO Objects based on waypoint, timestamp,speed and heading
    Stores BaseSpectra Objects based on waypoint and timestamp
    """

    def __init__(self):
        """
        RAO_Library constructor
        """
        self.RAO_dict = {}
        self.model_dict = {}

    def add_rao(self, wave_height, wave_period, speed, heading, wave_model):
        """
        Creates an rao with the given wave height and period, speed, and heading
        Adds the RAO to RAO_dict
        -------------------------------------
        Parameters
        ------------------------------------
        wave_height: float, rounded wave height
        wave_period: float, rounded wave heading
        speed: float, current vessel speed
        heading: float, relative heading between vessel and wave
        wave_model: BaseSpectra Object
        """
        rao = RAO(
            speed,
            heading,
            wave_model.Omegas,
            wave_model.AmpComponents,
            wave_model.Spectrum,
        )
        self.RAO_dict[(wave_height, wave_period, speed, heading)] = rao

    def add_wave_model(self, wave_height, wave_period, wave_model):
        """
        Adds wave_model to model_dict
        ------------------------------------
        Parameters
        ------------------------------------
        wave_height: float, rounded wave height
        wave_period: float, rounded wave heading
        wave_model: BaseSpectra Object
        """
        self.model_dict[(wave_height, wave_period)] = wave_model
