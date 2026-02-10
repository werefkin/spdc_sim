# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 11:40:43 2023

@author: r.zor
"""

from spdc_source import SPDC_source as spdc
from sellmeier_eqs import ref_ind_ktp_idl, ref_ind_ktp_si_pu

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


ppktp2 = spdc(
    0.65974,
    3.7,
    lambda_i_min,
    lambda_i_max,
    ref_ind_ktp_idl,
    ref_ind_ktp_si_pu,
    crystal_length=2.55,
    T=25)
ppktp2.plot_PSD_pp(pol_period=20.45)
