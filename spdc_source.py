import numpy as np
import matplotlib.pyplot as plt
import os


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

    def de_K_ap(self, poling_period_range):
        # poling_period_range in m over length of the crystal
        # phase mismatch becomes 2D in lambda and along the crystal length
        self.delta_k_base = 1 / self.c * (self.n_p * self.lambda_to_omega(self.lambda_p) - self.n_s * self.lambda_to_omega(
            self.lambda_s) - self.n_i * self.lambda_to_omega(self.lambda_i))
        self.Kz = - 2 * np.pi / poling_period_range
        self.delta_k_ap = self.delta_k_base[:, None] + self.Kz[None, :]
        return self.delta_k_ap

    def phase_ap(self, delta_k_ap=None):
        # accumulated phase along the crystal length for a given delta_k_ap(z) profile
        if delta_k_ap is None:
            delta_k_ap = self.delta_k_ap
        nn = delta_k_ap.shape[1]

        # uniform z-grid over the full crystal length
        self.z_ap = np.linspace(0, self.ll, nn)
        dz = self.z_ap[1] - self.z_ap[0]

        # cumulative trapezoidal integration along z
        self.phase = np.zeros_like(delta_k_ap)
        self.phase[:, 1:] = np.cumsum(
            0.5 * (delta_k_ap[:, 1:] + delta_k_ap[:, :-1]) * dz,
            axis=1
        )

        return self.phase

    def SPDC_spectrum_ap(self, poling_period_range=None, delta_k_ap=None, normalize=True):
        """
        Calculate SPDC spectrum for aperiodic/chirped poling:
            A(lambda) ~ int_0^L exp(i*phi(lambda,z)) dz
            PSD(lambda) = |A(lambda)|^2
        z-grid from 0 to self.ll.
        """
        if delta_k_ap is None:
            if poling_period_range is None:
                raise ValueError("Provide either poling_period_range or delta_k_ap")
            delta_k_ap = self.de_K_ap(poling_period_range)

        Nz = delta_k_ap.shape[1]
        z = np.linspace(0, self.ll, Nz)

        phase = self.phase_ap(delta_k_ap=delta_k_ap)

        integrand = np.exp(1j * phase)
        amp = np.trapz(integrand, z, axis=1)

        self.SPDC_PSD_ap = np.abs(amp) ** 2

        if normalize and np.max(self.SPDC_PSD_ap) > 0:
            self.SPDC_PSD_ap = self.SPDC_PSD_ap / np.max(self.SPDC_PSD_ap)

        return self.SPDC_PSD_ap

    def plot_PSD_ap(self, poling_period_range):
        Lambda_start_um = poling_period_range[0] * 1e6
        Lambda_end_um = poling_period_range[-1] * 1e6
        print(f"Plotting SPDC spectrum for poling period range: {Lambda_start_um:.2f} µm to {Lambda_end_um:.2f} µm")
        if np.isclose(Lambda_start_um, Lambda_end_um):
            le = (
                'T=' + str(self.temp) + '$^\\circ$; ' + r'$\Lambda$=' + str(round(Lambda_start_um, 2)) + ' µm; ' + r'$\lambda_p$=' + str(self.lambda_p * 1000) + ' nm; ' + 'L=' + str(self.ll * 1000) + ' mm'
            )
        else:
            le = (
                'T=' + str(self.temp) + '$^\\circ$; ' + r'$\Lambda(z)$=' + str(round(Lambda_start_um, 2)) + ' to ' + str(round(Lambda_end_um, 2)) + ' µm; ' + r'$\lambda_p$=' + str(self.lambda_p * 1000) + ' nm; ' + 'L=' + str(self.ll * 1000) + ' mm'
            )

        fig, ax1 = plt.subplots()
        ax1.plot(self.lambda_i, self.SPDC_PSD_ap, label=le)
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

        if np.isclose(Lambda_start_um, Lambda_end_um):
            name = (
                str(self.temp) + '_AP_' + str(round(Lambda_start_um, 2)) + '_' + str(self.lambda_p * 1000) + '_' + str(self.ll * 1000)
            )
        else:
            name = (
                str(self.temp) + '_AP_' + str(round(Lambda_start_um, 2)) + '_to_' + str(round(Lambda_end_um, 2)) + '_' + str(self.lambda_p * 1000) + '_' + str(self.ll * 1000)
            )

        os.makedirs("./output", exist_ok=True)
        plt.savefig('./output/' + name + '.png',
                    bbox_inches='tight',
                    transparent=True,
                    dpi=200)
        plt.show()

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
        os.makedirs("./output", exist_ok=True)
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
        os.makedirs("./output", exist_ok=True)
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

    def idler_to_signal(self, lambda_p, lambda_i):
        """
        Compute signal wavelength from pump and idler wavelengths
        """
        lambda_p = np.asarray(lambda_p, dtype=float)
        lambda_i = np.asarray(lambda_i, dtype=float)

        denom = 1.0 / lambda_p - 1.0 / lambda_i

        if np.any(denom <= 0):
            raise ValueError(
                "Non-physical combination for omega_p = omega_s + omega_i. "
                "You need lambda_i > lambda_p to get a positive signal frequency."
            )

        return 1.0 / denom
