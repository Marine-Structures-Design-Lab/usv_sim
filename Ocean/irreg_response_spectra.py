import numpy as np
import matplotlib.pyplot as plt


class Response_Spectra:
    def __init__(self, RAO, wave_spectrum, omegas):
        self.RAO = RAO
        self.wave_spectrum = wave_spectrum
        if len(RAO) != len(wave_spectrum):
            print("RAO length: ", len(RAO))
            print("wave_spectrum length: ", len(wave_spectrum))
            raise ValueError(
                "RAO and wave_spectrum must correspond to the same value of omegas (lengths must be equal))"
            )
        self.omegas = omegas

    def generate_response_spectra(self):
        """Generates the response spectra for a given wave spectrum and RAO.

        Returns:
            list of floats: response spectra for each frequency
        """
        response_spectra = np.zeros(len(self.omegas))
        for i in range(len(self.omegas)):
            response_spectra[i] = self.RAO[i] ** 2 * self.wave_spectrum[i]
        return response_spectra

    def generate_RMS(self, Hs, nonlinearity=False):
        """Generates the RMS value of the response spectra.

        Args:
            Hs (float): significant wave height
            nonlinearity (bool, optional): _description_. Defaults to False. Flag that determines whether to scale RMS values for sig wave heights greater than 3

        Returns:
            float: RMS value of the response spectra
        """
        Response_Spectra = self.generate_response_spectra()

        # Integrate the response spectra M0
        M0 = np.trapz(Response_Spectra, self.omegas, axis=0)
        RMS = np.sqrt(M0)
        if nonlinearity and Hs > 3:
            RMS = RMS * (0.25 * Hs / 3 + 1)
        return RMS

    def plot_response_spectrum(self, Hs):
        """Plots the response spectra.

        Args:
            response_spectra (list of floats): response for each frequency
        """
        response_spectra = self.generate_response_spectra()
        RMS = self.generate_RMS(Hs)
        # print(f"RMS: {RMS:}")
        plt.plot(self.omegas, response_spectra)
        plt.xlabel("Frequency (rad/s)")
        plt.ylabel("Response Spectra (m^2 s)")
        plt.title("Response Spectra")
        # Add RMS value to the plot as a text annotation
        plt.text(
            0.7,
            0.9,
            f"RMS: {RMS}",
            transform=plt.gca().transAxes,
            fontsize=12,
            bbox=dict(facecolor="white", alpha=0.7),
        )
        plt.show()
