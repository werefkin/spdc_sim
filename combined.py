# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 11:40:43 2023

@author: r.zor
"""

import numpy as np
import matplotlib.pyplot as plt

# SECTION: Sellmeier equations for the materials


def ref_ind_ln(lam_um, T=25):
    # Sellmeier equation for LB
    ref_ind = np.sqrt(1 +
                      2.6734 *
                      lam_um ** 2 /
                      (lam_um ** 2 -
                       0.01764) +
                      1.2290 *
                      lam_um ** 2 /
                      (lam_um ** 2 -
                       0.05914) +
                      12.614 *
                      lam_um ** 2 /
                      (lam_um ** 2 -
                          474.60))
    return ref_ind


def ref_ind_ktp(lam_um, T=25):
    # Sellmeier equation for KTP (Potassium titanyl phosphate)
    ref_ind = np.sqrt(1 +
                      1.71645 *
                      lam_um ** 2 /
                      (lam_um ** 2 -
                       0.013346) +
                      0.5924 *
                      lam_um ** 2 /
                      (lam_um ** 2 -
                       0.06503) +
                      0.3226 *
                      lam_um ** 2 /
                      (lam_um ** 2 -
                          67.1208) -
                      0.01133 *
                      lam_um ** 2)
    return ref_ind


def ref_ind_ktp_idl(lam_um, T=25):
    # Sellmeier equation for KTP (Potassium titanyl phosphate) - Katz et.al
    a1 = 1.00
    b1 = 1.71645
    b2 = 0.5924
    b3 = 0.3226
    c1 = 0.01346
    c2 = 0.06503
    c3 = 67.1208
    d1 = 0.01133

    ref_ind = np.sqrt(a1 +
                      b1 *
                      lam_um ** 2 /
                      (lam_um ** 2 -
                       c1) +
                      b2 *
                      lam_um ** 2 /
                      (lam_um ** 2 -
                       c2) +
                      b3 *
                      lam_um ** 2 /
                      (lam_um ** 2 -
                          c3) -
                      d1 *
                      lam_um ** 2)
    ref_ind = ref_ind + (T - 25) * (3.3920 + 0.5523 / lam_um - 1.7101 * lam_um + 0.3424 * lam_um ** 2) * 10**-5
    return ref_ind


def ref_ind_ktp_si_pu(lam_um, T=25):
    # Sellmeier equation for KTP (Potassium titanyl phosphate) - Fan et.al.
    a1 = 2.25411
    b1 = 1.06543
    c1 = 0.05486
    d1 = 0.02140
    ref_ind = np.sqrt(a1 + b1 / (1 - c1 / lam_um ** 2) - d1 * lam_um ** 2)
    ref_ind = ref_ind + (T - 25) * (-0.1897 + 3.6677 / lam_um - 2.9220 / lam_um ** 2 + 0.09221 / lam_um ** 3) * 10**-5
    return ref_ind


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
        self.c = 3 * 1e8
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
        self.SPDC_PSD = np.sinc(
            dK * ll / 2) ** 2  # Power spectral density
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
        plt.show()
        name = str(self.temp) + '_' + str(round(self.PP[self.ind] * 1e6, 2)) + '_' + str(self.lambda_p * 1000) + '_' + str(self.ll * 1000)
        plt.savefig('./output/' + name + '.png',
                    bbox_inches='tight',
                    transparent=True,
                    dpi=200)

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
        plt.show()
        name = str(self.temp) + '_' + str(pol_period * 1e6) + '_' + \
            str(self.lambda_p * 1000) + '_' + str(self.ll * 1000)
        plt.savefig('./output/' + name + '.png',
                    bbox_inches='tight',
                    transparent=True,
                    dpi=200)

    def plot_ref_indices(self):
        plt.figure()
        plt.plot(self.lambdas[:-1], self.n_g)
        plt.plot(self.lambdas, self.n)
        plt.ylabel('n, n$_g$')
        plt.xlabel('Wavelength (µm)')
        plt.show()


# Define idler range
lambda_i_min = 2.5  # um
lambda_i_max = 4.5  # um

# ppln = SPDC_source(0.785, 3.5, lambda_i_min, lambda_i_max, ref_ind_ln, ref_ind_ln, crystal_length=5)
# ppln.plot_PSD()
# ppln.plot_ref_indices()
# ppln.result()


ppktp = SPDC_source(
    0.65975,
    3.7,
    lambda_i_min,
    lambda_i_max,
    ref_ind_ktp_idl,
    ref_ind_ktp_si_pu,
    crystal_length=2.55,
    T=23)
# ppktp.plot_ref_indices()
# ppktp.result()
# ppktp.plot_PSD()
ppktp.plot_PSD_pp(pol_period=20.45)

# ppln = SPDC_source(0.785, 3.5, lambda_i_min, lambda_i_max, ref_ind_ln, ref_ind_ln, crystal_length=5)
# # ppln.plot_PSD()
# ppln.plot_PSD()

# # ppln.plot_ref_indices()
# ppln.result()
