# Simulation of (ultra-)broadband spontaneous-parametric down-conversion

A Python library for simulating the **(ultra-)broadband SPDC power spectral density (PSD)** profiles and estimating the **poling period** options (quasi-phase matching) under e.g. **group-velocity matching (GVM)** conditions.

The repository provides:
- `spdc_source.py`: a configurable `SPDC_source` class (inputs: refractive indices, pump wavelength, signal&idler spans; temperature)
- `example.py`: a minimal usage example (demonstrated for KTP crystal)
- `sellmeier_eqs.py`: Sellmeier equations used to compute refractive indices / dispersion

## Background

The implementation for periodic poling follows the modeling approach from (simpliest scenario):

Aron Vanselow, Paul Kaufmann, Helen M. Chrzanowski, and Sven Ramelow,  
**“Ultra-broadband SPDC for spectrally far separated photon pairs,”** *Optics Letters* **44**, 4638–4641 (2019).  
DOI: https://doi.org/10.1364/OL.44.004638

and for aperiodic structures generalized formalism for amplitude evoluton is used.

It is based on numerical solution of the integral for the idler amplitude:
```math
A_i(z, \omega) ∝ \int_{0}^{l} e^{i\phi(z', \omega)}\,dz'.
```

where

```math
\phi(z',\omega)=\int_{z_0}^{z'} \kappa(\xi,\omega)\,d\xi .
```
is the accumulated phase.

The SPDC spectral density is then computed as
$$
\mathrm{PSD}(\omega)\propto |A_i(\omega)|^2 .
$$

The methods are adapted from:

[1] Mathieu Charbonneau-Lefort, Bedros Afeyan, and M. M. Fejer,  
**“Optical parametric amplifiers using chirped quasi-phase-matching gratings I: practical design formulas,”** *Journal of the Optical Society of America B* **25**, 463-480 (2008).  
DOI: https://doi.org/10.1364/JOSAB.25.000463

[2] Martin M. Fejer, G.A. Magel, Dieter H. Jundt, and Robert L. Byer
**“Quasi-phase-matched second harmonic generation: tuning and tolerances,”** *IEEE Journal of Quantum Electronics* **28**(11), 2631-2654 (1992).  
DOI: https://doi.org/10.1109/3.161322

## Repository structure

- **`spdc_source.py`**  
  Defines an SPDC source model and routines to compute spectra and the optimal poling period for GVM / QPM settings.

- **`sellmeier_eqs.py`**  
  Includes Sellmeier equations for potassium titanyl phosphate **KTP** (demonstrated in the example) and also contains equations for **lithium niobate (LiNbO₃)**.

  To simulate other crystals, you’ll need to add the corresponding Sellmeier equations (refractive index vs. wavelength and, if applicable, temperature dependence).

- **`example.py`**  
  Demonstrates typical usage: defining an SPDC source, evaluating the PSD, and extracting an optimal poling period.
  Certain poling period (in the updated version including aperiodic structures) can be forced.

## Quick start

1. Clone the repository
2. Run the example:

```bash
python example.py
