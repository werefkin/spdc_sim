import numpy as np


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
