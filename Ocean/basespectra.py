import matplotlib.pyplot as plt
from math import cos, pi, exp, sqrt
from random import randint

import numpy as np

from matplotlib import pyplot as plt


class BaseSpectra:
    """
    Base class for all spectra classes.  Contains methods that every
    spectra class should implement
    """

    def __init__(self, name, phases=None, random_gen=None):
        """
        Base class consturctor

        Parameters
        ----------
        :param name: String
            Descriptive name for the spectrum
        :param random_gen : numpy random generator, optional
            Random number generator for time series if random phases are not
            supplied. The default is None, in which case new generator with
            seed 48109 is started.

        Returns
        -------
        None.

        """

        if random_gen is None:
            self.generator = np.random.default_rng(48109)
        else:
            self.generator = random_gen
        self.phases = phases
        self.name = name
        self.AmpComponents = None  # This will hold future amplitude components
        self.Omegas = None  # Hold corresponding omegas to AmpComponents
        self.NumComp = None  # Number of spectral components

    def timeSeries(self, times, phases=None):
        """
        Calculates a time series record of the specified spectra

        Parameters
        ----------
        :param times: 1 x t numpy array
            Time, in seconds, at which the wave elevation should be calculated
            at
        :param phases: numpy array, optional
            Random phase angles for the spectral component. If none, and no
            phases have been set yet on the object, random phases will be used

        Returns
        -------
        1 x t numpy array of wave heights
        """

        # First check phases
        if phases is None and self.phases is None:
            phases = self.generator.random(len(self.AmpComponents)) * 2 * pi
            self.phases = phases
        else:
            if len(phases) != len(self.AmpComponents):
                raise Exception(
                    "Phase/Amp Components of different length" + " in BaseSpectra"
                )

        # Make an output array
        heights = np.empty_like(times)

        # Make an intermediate array
        cos_argument = np.empty_like(phases)

        # Loop over all the time series - use enumerate to get index and time
        for index, t in enumerate(times):
            cos_argument = self.Omegas * t + phases
            cos_argument = np.cos(cos_argument)

            # Use dot product as quick way of doing summation
            heights[index] = np.dot(cos_argument, self.AmpComponents)

        return heights

    def heightVel(self, time):
        """
        Returns the height and surface velocity of spectra at a given time

        Parameters
        ----------
        :param time: float
            Time to calculate spectra

        Returns
        -------
        Tuple of (height, velocity) in units spectra configured in

        """
        # First check phases
        if self.phases is None:
            self.phases = self.generator.random(len(self.AmpComponents)) * 2 * pi
        else:
            if len(self.phases) != len(self.AmpComponents):
                raise Exception(
                    "Phase/Amp Components of different length" + " in BaseSpectra"
                )

        # Interior argument is the same for sin/cos
        trans_argument = np.empty_like(self.phases)
        trans_argument = self.Omegas * time + self.phases

        # Height
        costerm = np.cos(trans_argument)
        height = np.dot(costerm, self.AmpComponents)

        # Do the velocity calculation = -omega*sin(omega*t+phase)
        sinterm = -self.Omegas * np.sin(trans_argument)
        velocity = np.dot(sinterm, self.AmpComponents)

        return (height, velocity)


class JONSWAP(BaseSpectra):
    def __init__(self, name, H13, Tp, Gamma=3.3, wMin=0.001, wMax=5, nComponents=500):
        """
        Builds a basic JonSWAP spectrum

        Parameters
        ----------
        :param name: String
            Description of the specta
        :param H13: Float
            Significant wave height of the spectra
        :param Tp: Float
            Peak period of the spectra
        :param  Gamma: Float
            Peakedness factor, defaults to 3.3
        :param wMin: Float
            Miniumum frequency, in rad/sec defaults to 0.01.
        :param wMax: Float
            Maximum frequency, in rad/sec, defaults to 3.0
        :param nComponents: Int
            Number of components to discretize spectrum into.  Defaults to 500

        Returns
        -------
        None.

        """
        super().__init__(name)

        self.NumComp = nComponents

        # Discretize the spectra

        # Set up a range of omegas
        self.Omegas = np.linspace(wMin, wMax, self.NumComp)
        deltaOmega = self.Omegas[1] - self.Omegas[0]

        # Spectral output variables
        self.Spectrum = np.empty_like(self.Omegas)
        self.AmpComponents = np.empty_like(self.Omegas)

        # Calculate peak frequency
        omega_peak = 2 * pi / Tp

        # For each frequency, get spectral height and amplitude component
        for index, Omega in enumerate(self.Omegas):
            # This is 5.126 in Journee Offshore Hydromechanics
            if Omega < omega_peak:
                sigma = 0.07
            else:
                sigma - 0.09
            A = exp(-(((Omega / omega_peak - 1) / (sigma * 2 ** (0.5))) ** 2.0))

            # Calculate spectral height
            self.Spectrum[index] = (
                320.0
                * H13 ** (2.0)
                / Tp ** (4.0)
                * Omega ** (-5.0)
                * exp(-1950 / Tp ** (4.0) * Omega ** (-4.0))
                * Gamma ** (A)
            )

            # Calculate Amplitude coefficient - 5.129 in Journee
            self.AmpComponents[index] = sqrt(2.0) * (
                self.Spectrum[index] * deltaOmega
            ) ** (0.5)


class JessicaSpectra:
    #     '''
    #     Base class for all spectra classes.  Contains methods that every
    #     spectra class should implement
    #     '''
    def __init__(self, name):
        self.name = name

    # self, time, numcomponents, phases=None
    def timeSeries():
        np_array_time = []
        np_array_nc = []
        np_array_phase = []
        np_array_wave_elev = []
        w = 1
        x = 0
        name = str(input("Please enter the name of the spectra: "))
        elem = int(
            input(
                "Please enter the number of times you would like to calculate"
                " wave elevation at: "
            )
        )
        print("Enter each time: ")
        for i in range(int(elem)):
            x = x + 1
            time = float(input("Time: "))
            np_array_time.append(time)
        num_comp = int(
            input(
                "Please input the number of components you would like to use to"
                " reconstruct the spectra: "
            )
        )
        for i in range(int(num_comp)):
            numcomponents = input("Components:")
            np_array_nc.append(numcomponents)
        phase_ang = np.radians(
            input(
                "Please input the phase angle(in rad) for each component. "
                "If none, type 0"
            )
        )
        if phase_ang == 0:
            phase_ang = np.radians(randint(1, 90))
        np_array_phase.append(phase_ang)
        for i in range(elem):
            ct = cos(
                np.astype(np_array_nc[i]) * x
                - (w * np.astype(np_array_time[i]))
                + np.astype(np_array_phase[i])
            )
            np_array_wave_elev.append(ct)
        d = 0
        for i in range(elem):
            d = d + 1
            plt.plot(np_array_wave_elev[i], d)
        return np_array_wave_elev


class Bretschneider(BaseSpectra):
    def __init__(self, name, H13, T1, wMin=0.01, wMax=3.0, numComponents=500):
        """
        Builds a basic Bretshneider spectrum

        Parameters
        ----------
        :param name: string
            Descriptive name for the spectrum
        :param H13: float
            Significant wave height for the spectrum
        :param T1: float
            Mean centroid wave period for the spectrum
        :param wMin: float
            Minimum frequency in rad/sec, defaults to 0.01
        :param wMax: float
            Maximum frequency in rad/sec, defaults to 3.0
        :param numComponents: int
            Number of components to discretize spectrum into

        Returns
        -------
        None
        """
        super().__init__(name)
        self.NumComp = numComponents
        self.Omegas = np.linspace(wMin, wMax, numComponents)
        self.Spectrum = np.empty_like(self.Omegas)
        self.AmpComponents = np.empty_like(self.Omegas)
        deltaOmega = self.Omegas[1] - self.Omegas[0]
        # For each frequency calculate spectral height and amplitude component
        for index, Omega in enumerate(self.Omegas):
            # Journee equation 5.123 - Spectral height
            self.Spectrum[index] = (
                ((173 * H13 ** (2)) / T1 ** (4))
                * Omega ** (-5)
                * exp((-692 / T1 ** (4)) * Omega ** (-4))
            )
            # Journee equation 5.129 - Amplitude component
            self.AmpComponents[index] = sqrt(2.0) * (
                self.Spectrum[index] * deltaOmega
            ) ** (0.5)


# test plotting time series
if __name__ == "__main__":
    model = Bretschneider("test", 4, 10)
    heights = model.timeSeries(np.arange(0, 1000) / 10)
    plt.plot(heights)
    plt.savefig("test.png")
