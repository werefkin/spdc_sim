import numpy as np
import matplotlib.pyplot as plt


class SPDC_source:
    def __init__(
            self,
            lambda_pump,
            center_wavelength,
            lambda_idler_min,
            lambda_idler_max,
            ref_ind_function_idler,
            ref_ind_function_signal_pump,
            N=1024,
            crystal_length=2,
            T=25):
        """
        Parameters
        ----------
        lambda_pump : float
            Pump wavelength in um. Sets the pump frequency used for
            energy conservation and phase-matching calculations.

        center_wavelength : float
            Center wavelength in um to which the Delta_k = 0 is forced (to calculate a poling period).

        lambda_idler_min : float
            Minimum idler wavelength to include in the simulation window
            (um).

        lambda_idler_max : float
            Maximum idler wavelength to include in the simulation window
            (um).

        ref_ind_function_idler : callable
            Refractive-index (Sellmeier) function for the idler.
            Expected signature should be something like:
                n_i = ref_ind_function_idler(lambda_um, T_C)
            returning the refractive index at wavelength `lambda_um` and (optinal) temperature `T_C`.
            (If your Sellmeier functions do not use temperature, they may ignore `T_C`.)

        ref_ind_function_signal_pump : callable
            Refractive-index (Sellmeier) function used for signal and pump
            Expected signature similar to:
                n_sp = ref_ind_function_signal_pump(lambda, T_C)

        ref_ind_function_idler and ref_ind_function_signal_pump can be the same if well defined.
        
        N : int, optional (default: 1024)
            Number of sampling points in the spectral grid.

        crystal_length : float, (default: 2)
            Nonlinear crystal length in mm.

        T : float, optional (default: 25)
            Crystal temperature in degrees Celsius. Used by temperature-dependent (if they are)
            Sellmeier equations (and thus affects phase matching / optimal poling period).
        """
        self.c = 299792458
        self.lambda_p = lambda_pump
        self.center_wavelength = center_wavelength
        self.lambda_i = np.linspace(lambda_idler_min, lambda_idler_max, N)
        # in um, just for visualization of n
        self.lambdas = np.linspace(0.5, 5, N)
        self.ll = crystal_length * 1e-3
        self.temp = T
        self.ref_ind = ref_ind_function_signal_pump
        self.ref_ind_idl = ref_ind_function_idler

        self.omega = self.lambda_to_omega(self.lambdas)

        self.lambda_s = self.lambda_i * self.lambda_p / \
            (self.lambda_i - self.lambda_p)  # um

        self.n_p = self.ref_ind(self.lambda_p, self.temp)
        self.n_i = self.ref_ind_idl(self.lambda_i, self.temp)
        self.n_s = self.ref_ind(self.lambda_s, self.temp)
        self.n_g = self.ref_ind(self.lambdas[:-1]) + self.omega[:-1] * (np.diff(
            self.ref_ind(self.lambdas)) / np.diff(self.omega))  # Group index to display
        self.n = self.ref_ind(self.lambdas)

        self.PP = 2 * np.pi * self.c * (self.n_p * self.lambda_to_omega(self.lambda_p) - self.n_s * self.lambda_to_omega(
            self.lambda_s) - self.n_i * self.lambda_to_omega(self.lambda_i)) ** -1  # Optimal poling periods vector
        self.ind = np.where(self.lambda_i < self.center_wavelength)[0][-1]
        self.delta_k = 1 / self.c * (self.n_p * self.lambda_to_omega(self.lambda_p) - self.n_s * self.lambda_to_omega(
            self.lambda_s) - self.n_i * self.lambda_to_omega(self.lambda_i)) - 2 * np.pi / self.PP[self.ind]

    def de_K(self, def_poling_period):
        self.delta_k = 1 / self.c * (self.n_p * self.lambda_to_omega(self.lambda_p) - self.n_s * self.lambda_to_omega(
            self.lambda_s) - self.n_i * self.lambda_to_omega(self.lambda_i)) - 2 * np.pi / def_poling_period
        return self.delta_k

    def result(self):
        print('Optimal polling period for ' + str(self.center_wavelength) + 'um wavelength is: ' + str(round(self.PP[self.ind] * 1e6, 2)) + 'um')

    def lambda_to_omega(self, lambd):
        omeg = self.c * 2 * np.pi / (lambd * 1e-6)
        return omeg

    def SPDC_spectrum(self, dK=None, ll=None):
        if dK is None:
            dK = self.delta_k
        if ll is None:
            ll = self.ll
        x = dK * ll / 2
        self.SPDC_PSD = (np.sin(x) / x) ** 2
        return self.SPDC_PSD

    def plot_PSD(self):
        self.de_K(self.PP[self.ind])
        le = 'T=' + str(self.temp) + '$^\\circ$; ' + r'$\Lambda$=' + str(round(self.PP[self.ind] * 1e6, 2)) + ' µm; ' + r'$\lambda_p$=' + str(
            self.lambda_p * 1000) + ' nm; ' + 'L=' + str(self.ll * 1000) + ' mm'
        fig, ax1 = plt.subplots()
        ax1.plot(self.lambda_i, self.SPDC_spectrum(), label=le)
        ax1.set_xlabel('Wavelength Idler (µm)')
        ax1.set_ylabel('PSD (a.u.)')
        ax1.set_ylim([0, None])
        self.xtick_positions = plt.gca().get_xticks()
        ax1.legend()
        ax2 = ax1.twiny()
        ax2.set_xlim(ax1.get_xlim())
        ax2.set_xlabel('Wavelength Signal (µm)')
        ax2.set_xticks(self.xtick_positions)
        ax2.set_xbound(ax1.get_xbound())
        ax2.set_xticklabels(np.round(
            self.xtick_positions * self.lambda_p / (self.xtick_positions - self.lambda_p), 3))
        ax1.set_ylim(0, 1.2)
        name = str(self.temp) + '_' + str(round(self.PP[self.ind] * 1e6, 2)) + '_' + str(self.lambda_p * 1000) + '_' + str(self.ll * 1000)
        plt.savefig('./output/' + name + '.png',
                    bbox_inches='tight',
                    transparent=True,
                    dpi=200)
        plt.show()

    def plot_PSD_pp(self, pol_period):
        pol_period = pol_period * 1e-6
        dK_lo = self.de_K(pol_period)
        le = 'T=' + str(self.temp) + '$^\\circ$; ' + r'$\Lambda$=' + str(round(pol_period * 1e6, 2)) + \
            ' µm; ' + r'$\lambda_p$=' + str(self.lambda_p * 1000) + ' nm; ' + 'L=' + str(self.ll * 1000) + ' mm'

        fig, ax1 = plt.subplots()
        ax1.plot(self.lambda_i, self.SPDC_spectrum(dK=dK_lo), label=le)
        ax1.set_xlabel('Wavelength Idler (µm)')
        ax1.set_ylabel('PSD (a.u.)')
        ax1.set_ylim([0, None])
        self.xtick_positions = plt.gca().get_xticks()
        ax1.legend()
        ax2 = ax1.twiny()
        ax2.set_xlim(ax1.get_xlim())
        ax2.set_xlabel('Wavelength Signal (µm)')
        ax2.set_xticks(self.xtick_positions)
        ax2.set_xbound(ax1.get_xbound())
        ax2.set_xticklabels(np.round(
            self.xtick_positions * self.lambda_p / (self.xtick_positions - self.lambda_p), 3))
        ax1.set_ylim(0, 1.2)
        name = str(self.temp) + '_' + str(pol_period * 1e6) + '_' + \
            str(self.lambda_p * 1000) + '_' + str(self.ll * 1000)
        plt.savefig('./output/' + name + '.png',
                    bbox_inches='tight',
                    transparent=True,
                    dpi=200)
        plt.show()

    def plot_ref_indices(self):
        plt.figure()
        plt.plot(self.lambdas[:-1], self.n_g)
        plt.plot(self.lambdas, self.n)
        plt.ylabel('n, n$_g$')
        plt.xlabel('Wavelength (µm)')
        plt.show()
