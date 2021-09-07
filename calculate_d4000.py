###############################################################################
#
#   File:       calculate_d4000.py
#   Author:     Henry Zovaro
#   Email:      henry.zovaro@anu.edu.au
#
#   Description:
#   Compute the Dn4000Å break strength for the S7 sample.
#
#   Copyright (C) 2021 Henry Zovaro
#
###############################################################################
#!/usr/bin/env python
from __future__ import print_function
import sys, os

import matplotlib
# matplotlib.use("Agg")

from time import time

from astropy.io import fits
from astroquery.ned import Ned
from astroquery.irsa_dust import IrsaDust
from scipy import constants, ndimage
import numpy as np
import extinction
from itertools import product
import multiprocessing
import pandas as pd

from cosmocalc import get_dist

from IPython.core.debugger import Tracer

##############################################################################
# Plotting settings
##############################################################################
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
matplotlib.rc("font", size=14)
matplotlib.rc("text", usetex=False)
matplotlib.rc("font", **{"family": "serif"})
matplotlib.rc("image", interpolation="nearest")
matplotlib.rc("image", origin="lower")
plt.close("all")
plt.ion()

##############################################################################
# For interactive execution
##############################################################################
def hit_key_to_continue():
    key = input("Hit a key to continue or q to quit...")
    if key == 'q':
        sys.exit()
    return

##############################################################################
# Paths and filenames
##############################################################################
assert "S7_DIR" in os.environ, 'S7_DIR environment variable is not defined! Make sure it is defined in your .bashrc file: export S7_DIR="/path/to/s7/data/"'
data_dir = os.environ["S7_DIR"]
input_fits_path =  os.path.join(data_dir, "0_Cubes")  # Path to S7 data cubes
for path in [input_fits_path]:
    assert os.path.exists(path), "Directory {} does not exist!".format(path) 

# All S7 galaxies
gals = [fname.split("_B.fits")[0] for fname in os.listdir(input_fits_path) if fname.endswith("_B.fits") and not fname.startswith(".")]
df_output  = pd.DataFrame(index=gals)

# Figure window for plotting
plotit = True
if plotit:
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))

# Empty dataframe for saving results
df = pd.DataFrame(index=gals)

for obj_name in gals:
    print(f"Processing {obj_name}...")

    # Name of input FITS file
    input_fits_fname = "{}_B.fits".format(obj_name)
    assert os.path.exists(os.path.join(input_fits_path, input_fits_fname)), "File {} does not exist!".format(input_fits_fname)

    ##############################################################################
    # Ojbect information
    ##############################################################################
    # load the S7 catalogue
    df_metadata = pd.read_csv(os.path.join(data_dir, "s7_metadata.csv"), comment="$")
    df_metadata = df_metadata.set_index("S7_Name")
    r = df_metadata.loc[obj_name, "HL_Re"]   # radius of aperture in arcsec
    z = df_metadata.loc[obj_name, "S7_z"]  # redshift
    x_0 = df_metadata.loc[obj_name, "S7_nucleus_index_x"]
    y_0 = df_metadata.loc[obj_name, "S7_nucleus_index_y"]
    D_A_Mpc, D_L_Mpc = get_dist(z, H0=70.0, WM=0.3)

    # Extinction
    t = IrsaDust.get_query_table(obj_name, radius="2deg")
    A_V_Gal = t["ext SandF ref"] * 3.1  # S&F 2011 A_V (= E(B-V) * 3.1)

    ##############################################################################
    # Open the data cube containing the galaxy spectra
    ##############################################################################
    # Load FITS file containing WiFeS data
    hdu = fits.open(os.path.join(input_fits_path, input_fits_fname))
    data_cube = hdu[0].data # TODO: check if we need to crop at all
    var_cube = hdu[1].data
    N_lambda, nrows, ncols = data_cube.shape

    # NaN out the bottom couple of rows
    data_cube[:, :2, :] = np.nan
    var_cube[:, :2, :] = np.nan

    # Wavelength information
    lambda_start_A = hdu[0].header["CRVAL3"]
    lambda_end_A = hdu[0].header["CRVAL3"] + \
        hdu[0].header["CDELT3"] * (N_lambda - 1)
    dlambda_A = hdu[0].header["CDELT3"]
    lambda_vals_A = np.arange(
        start=lambda_start_A, stop=lambda_end_A + dlambda_A, step=dlambda_A)
    lambda_vals_A = lambda_vals_A[:data_cube.shape[0]]

    hdu.close()

    ##############################################################################
    # Correct for galactic extinction
    ##############################################################################
    A_vals = extinction.fm07(lambda_vals_A, a_v=A_V_Gal)
    data_cube *= 10**(0.4 * A_vals[:, None, None])
    var_cube *= 10**(2 * 0.4 * A_vals[:, None, None])

    ##############################################################################
    # Extract the aperture spectrum
    ##############################################################################
    yy, xx = np.meshgrid(range(data_cube.shape[1]), range(
        data_cube.shape[2]), indexing='ij')

    # Mask out spaxels beyond the given radius
    aperture = (xx - x_0)**2 + (yy - y_0)**2 < r**2

    data_cube[:, ~aperture] = np.nan
    var_cube[:, ~aperture] = np.nan

    # Extract spectrum
    spec = np.nansum(np.nansum(data_cube, axis=1), axis=1) * 4 * np.pi * (D_L_Mpc * 1e6 * 3.086e18)**2  # Needs to be in units of erg/s/A
    spec_err = np.sqrt(np.nansum(np.nansum(var_cube, axis=1), axis=1)) * 4 * np.pi * (D_L_Mpc * 1e6 * 3.086e18)**2  # Needs to be in units of erg/s/A
    spec_var = spec_err**2

    ##############################################################################
    # Compute the Dn4000Å break strength
    ##############################################################################
    # Compute the D4000Å break
    # Definition from Balogh+1999 (see here: https://arxiv.org/pdf/1611.07050.pdf, page 3)
    start_b_idx = np.nanargmin(np.abs(lambda_vals_A / (1 + z) - 3850))
    stop_b_idx = np.nanargmin(np.abs(lambda_vals_A / (1 + z) - 3950))
    start_r_idx = np.nanargmin(np.abs(lambda_vals_A / (1 + z) - 4000))
    stop_r_idx = np.nanargmin(np.abs(lambda_vals_A / (1 + z) - 4100))
    N_b = stop_b_idx - start_b_idx
    N_r = stop_r_idx - start_r_idx

    # Convert datacube & variance cubes to units of F_nu
    spec_Hz = spec * lambda_vals_A**2 / (constants.c * 1e10)
    spec_var_Hz2 = spec_var * (lambda_vals_A[:, None, None]**2 / (constants.c * 1e10))**2

    num = np.nanmean(spec_Hz[start_r_idx:stop_r_idx], axis=0)
    denom = np.nanmean(spec_Hz[start_b_idx:stop_b_idx], axis=0)
    err_num = 1 / N_r * np.sqrt(np.nansum(spec_var_Hz2[start_r_idx:stop_r_idx]))
    err_denom = 1 / N_b * np.sqrt(np.nansum(spec_var_Hz2[start_b_idx:stop_b_idx]))

    d4000 = num / denom
    d4000_err = d4000 * np.sqrt((err_num / num)**2 + (err_denom / denom)**2)

    # Plot
    if plotit:
        ax.clear()
        ax.plot(lambda_vals_A, spec, "k")
        ax.axvspan(lambda_vals_A[start_b_idx], lambda_vals_A[stop_b_idx], alpha=0.5, facecolor="orange")
        ax.axvspan(lambda_vals_A[start_r_idx], lambda_vals_A[stop_r_idx], alpha=0.5, facecolor="orange")
        ax.set_xlabel("Rest-frame wavelength (\AA)")
        ax.set_ylabel(r"$F_{\lambda}(\lambda)$")
        ax.set_title(obj_name)
        ax.text(x=0.1, y=0.9, s=r"$\rm D_n4000$\AA break strength = $%.2f \pm %.2f$" % (d4000, d4000_err), transform=ax.transAxes)
        fig.canvas.draw()
        hit_key_to_continue()

    # Save to dataframe
    df.loc[obj_name, "D4000"] = d4000
    df.loc[obj_name, "D4000 error"] = d4000_err

# Save to file
df.to_csv(os.path.join(data_dir, "S7_D4000.csv"))
