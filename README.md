# Simulation of broadband SPDC PSD (group velocity matching)

A small Python library for simulating the **broadband SPDC power spectral density (PSD)** and estimating the **optimal poling period** (quasi-phase matching) under **group-velocity matching (GVM)** conditions.

The repository provides:
- `spdc_source.py`: a configurable `SPDC_source` class (inputs: refractive indices, pump wavelength, signal&idler spans; temperature)
- `example.py`: a minimal usage example
- `sellmeier_eqs.py`: Sellmeier equations used to compute refractive indices / dispersion

## Background

This implementation follows the modeling approach from:

Aron Vanselow, Paul Kaufmann, Helen M. Chrzanowski, and Sven Ramelow,  
**“Ultra-broadband SPDC for spectrally far separated photon pairs,”** *Optics Letters* **44**, 4638–4641 (2019).  
DOI: https://doi.org/10.1364/OL.44.004638

## Repository structure

- **`spdc_source.py`**  
  Defines an SPDC source model and routines to compute spectra and the optimal poling period for GVM / QPM settings.

- **`sellmeier_eqs.py`**  
  Includes Sellmeier equations for potassium titanyl phosphate **KTP** (demonstrated in the example) and also contains equations for **lithium niobate (LiNbO₃)**.

  To simulate other crystals, you’ll need to add the corresponding Sellmeier equations (refractive index vs. wavelength and, if applicable, temperature dependence).

- **`example.py`**  
  Demonstrates typical usage: defining an SPDC source, evaluating the PSD, and extracting an optimal poling period.
  Certain poling period can be forced.

## Quick start

1. Clone the repository
2. Run the example:

```bash
python example.py
