# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 11:40:43 2023

@author: r.zor
"""

from spdc_source import SPDC_source as spdc
from sellmeier_eqs import ref_ind_ktp_idl, ref_ind_ktp_si_pu
import numpy as np
import matplotlib.pyplot as plt

# Define idler range
lambda_i_min = 2.5  # um
lambda_i_max = 5.5  # um
pump_wavelength = 0.65974  # um
crystal_length = 5  # mm

# ppln = spdc(0.785, 3.5, lambda_i_min, lambda_i_max, ref_ind_ln, ref_ind_ln, crystal_length=5)
# ppln.plot_PSD()
# ppln.plot_ref_indices()
# ppln.result()

ppktp = spdc(
    pump_wavelength,
    3.7,
    lambda_i_min,
    lambda_i_max,
    ref_ind_ktp_idl,
    ref_ind_ktp_si_pu,
    crystal_length=crystal_length,
    T=23)

ppktp.plot_PSD_pp(pol_period=20.45)


# Aperioddic case
PN = 16384
p_sta = 19.5
P_end = 20.6
p_range = np.linspace(p_sta, P_end, PN)

# Fun with chirps
# # quadratic chirp
# p_mid = 19.51      # midd
# dp = 20.56 - 19.51      # edges

# u = np.linspace(0, 1, PN)   # center = 0, edges = ±1
# p_range = p_mid + dp * u**2
# Sinusoidal chirp
# p_range = p_mid + np.sin(10 * u * np.pi - np.pi / 2) * dp * 10

plt.figure()
plt.plot(np.linspace(0, crystal_length, PN), p_range)
plt.xlabel(r'Position along the crystal $z$ (mm)')
plt.ylabel('Poling period (µm)')
plt.show()


ppktp.SPDC_spectrum_ap(poling_period_range=p_range * 1e-6)
sim_spe = ppktp.SPDC_spectrum_ap(poling_period_range=p_range * 1e-6)
sim_spe_pp = ppktp.SPDC_spectrum()

# Plot of general (e.g. chirped) vs periodic poling (sinc^2 solution)
xxi = np.linspace(lambda_i_min, lambda_i_max, len(sim_spe))
plt.figure()
plt.plot(xxi, sim_spe, label='Generalized solution')
plt.plot(xxi, sim_spe_pp, label='Periodic poling')
plt.legend()
plt.xlabel('Signal wavelength (µm)')
plt.ylabel('SPDC intensity (a.u.)')
plt.show()
