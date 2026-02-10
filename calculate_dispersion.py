# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 11:40:43 2023

@author: r.zor
"""

from spdc_source import SPDC_source as spdc
from sellmeier_eqs import ref_ind_ktp_idl, ref_ind_ktp_si_pu
import matplotlib.pyplot as plt
import numpy as np

# Define idler range
lambda_i_min = 2.5  # um
lambda_i_max = 4.5  # um

# ppln = spdc(0.785, 3.5, lambda_i_min, lambda_i_max, ref_ind_ln, ref_ind_ln, crystal_length=5)
# ppln.plot_PSD()
# ppln.plot_ref_indices()
# ppln.result()

ppktp = spdc(
    0.65974,
    3.7,
    lambda_i_min,
    lambda_i_max,
    ref_ind_ktp_idl,
    ref_ind_ktp_si_pu,
    crystal_length=2.55,
    T=23)

ppktp.plot_PSD_pp(pol_period=20.45)

# calculate necessary units
ni = ppktp.n_i
ns = ppktp.n_s
lam_i = ppktp.lambda_i
lam_s = ppktp.lambda_s
omega_i = ppktp.lambda_to_omega(lam_i)
omega_s = ppktp.lambda_to_omega(lam_s)


fig, ax = plt.subplots()
ax.plot(lam_i, ni)
ax.plot(lam_s, ns)


# calculate group index

ng_i = ni[:-1] + omega_i[:-1] * np.diff(ni) / np.diff(omega_i)
ng_s = ns[:-1] + omega_s[:-1] * np.diff(ns) / np.diff(omega_s)

fig, ax = plt.subplots()
ax.plot(lam_i[:-1], ng_i)
ax.plot(lam_s[:-1], ng_s)


# delay = ng/c*length, length = 0.00255

length = 0.00255
del_i = ng_i / (3 * 1e8) * length
del_s = ng_s / (3 * 1e8) * length

# if negative --> signal has higher delay --> slower
diff = (del_i - del_s) * 1e15

fig, ax = plt.subplots()
ax.plot(lam_s[400:900], diff[400:900])
# ax.set_xlim(0.78,0.82)
ax.set_xlabel('Wavelengths / nm')
ax.set_ylabel('Delay / fs')


# calculate group delay dispersion

# gdd_i = np.diff(del_i)/np.diff(omega_i)
gdd = np.diff((del_i - del_s)) / np.diff(omega_s[:-1]) * 1e30  # fs^2

fig, ax = plt.subplots()
ax.plot(lam_s[400:900], gdd[400:900])

plt.show()
