"""This file generates a time series response off a frequency spectrum.
"""

import numpy as np
import matplotlib.pyplot as plt


class Time_Series:
    def __init__(self, spectrum, omegas):
        """Initializes the Wave_Spectra class.

        Args:
            Spectrum (list of floats): spectrum response for each frequency
            omegas (list of floats): wave frequencies in radians per second
        """
        self.spectrum = spectrum
        self.omegas = omegas
        self.delta_omega = (
            self.omegas[1] - self.omegas[0]
        )  # change in omega between each spectrum data point. Radians per second
        self.eps = np.random.uniform(
            0, 2 * np.pi, len(self.omegas)
        )  # random phase angles are used and adequate

    def batch_response(self, t, x):
        """This function generates a batch response of response amplitudes for all frequencies for a given time and position.
            Uses batch multiplication of arrays to reduce computation time.

        Args:
            t (float): time in seconds
            x (float): meters in direction of wave propagation

        Returns:
            array of floats: amplitudes at a given time and position for each frequency
        """
        amp = 2 * np.sqrt(
            np.array(self.spectrum) * self.delta_omega
        )  # amplitude of each frequency component
        k = self.omegas**2 / 9.81
        amp_t = amp * np.cos(k * x - self.omegas * t + self.eps)
        return amp_t

    def generate_time_series(self, time_range, x):
        """Superimposes the batch responses to generate the overall waves for a given time range and position.

        Args:
            time_range (list of floats): time range in seconds to generate the wave spectra
            x (float): meters in direction of wave propagation

        Returns:
            list of floats: amplitudes for each time in the time range
        """
        # initialize the response spectra
        amps = np.zeros(len(time_range))
        for i in range(len(time_range)):
            # superimpose the response spectra
            amps[i] = np.sum(self.batch_response(time_range[i], x))
        return amps

    def plot(self, time_range, amps):
        """Plots the spectrum in the time domain.

        Args:
            time_range (list of floats): time range in seconds to generate the wave spectra
            amps (numpy array of floats): amplitudes for each time in the time range
        """
        plt.plot(time_range, amps)
        plt.ylabel("Amplitude (m)")
        plt.xlabel("Time (s)")
        plt.title("Time Domain Response")
        plt.show()
