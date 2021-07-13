#!/usr/bin/env python
from __future__ import print_function
import sys

import os
from astropy.io import fits
from astropy.wcs import WCS
from astroquery.ned import Ned

import matplotlib.pyplot as plt

from itertools import product

from scipy import ndimage, interpolate

import numpy as np

from IPython.core.debugger import Tracer

##############################################################################
# Open both FITS files
##############################################################################
data_dir = "/priv/meggs3/u5708159/S7" 
fig_path = os.path.join(data_dir, "figs/")
fits_path = os.path.join(data_dir, "0_Cubes")
obj_name = sys.argv[1]

# get object information from NED
n = Ned.query_object(obj_name)
z = n["Redshift"].data[0]  # Initial estimate of the galaxy redshift

fname_b = os.path.join(fits_path, "{}_B.fits".format(obj_name))
fname_r = os.path.join(fits_path, "{}_R.fits".format(obj_name))

hdulist_B3000 = fits.open(fname_b)
datacube_b = hdulist_B3000[0].data
varcube_b = hdulist_B3000[1].data

hdulist_R7000 = fits.open(fname_r)
datacube_r = hdulist_R7000[0].data
varcube_r = hdulist_R7000[1].data

# Get the approx. centre coordinates of the galaxy
y_0_px, x_0_px = np.unravel_index(np.nanargmax(np.nanmean(datacube_r, axis=0)), shape=datacube_r.shape[1:])

# Spatial information
N_lambda_b, nrows, ncols = datacube_b.shape
wcs = WCS(hdulist_B3000[0].header).dropaxis(2)

# Wavelength information
lambda_0_b_A = hdulist_B3000[0].header["CRVAL3"]
lambda_end_b_A = hdulist_B3000[0].header["CRVAL3"] + \
    hdulist_B3000[0].header["CDELT3"] * (N_lambda_b)
dlambda_b_A = hdulist_B3000[0].header["CDELT3"]
lambda_vals_b_A = np.arange(
    start=lambda_0_b_A, stop=lambda_end_b_A, step=dlambda_b_A)

N_lambda_r, nrows, ncols = datacube_r.shape
lambda_0_r_A = hdulist_R7000[0].header["CRVAL3"]
lambda_end_r_A = hdulist_R7000[0].header["CRVAL3"] + \
    hdulist_R7000[0].header["CDELT3"] * (N_lambda_r)
dlambda_r_A = hdulist_R7000[0].header["CDELT3"]
lambda_vals_r_A = np.arange(
    start=lambda_0_r_A, stop=lambda_end_r_A, step=dlambda_r_A)

# Instrumental resolution
FWHM_inst_b_A = 1.4  # as measured using sky lines in the b grating
sigma_inst_b_A = FWHM_inst_b_A / (2 * np.sqrt(2 * np.log(2)))

FWHM_inst_r_A = 0.9
sigma_inst_r_A = FWHM_inst_r_A / (2 * np.sqrt(2 * np.log(2)))

##############################################################################
# Spectrally convolve the R7000 to the same spectral resolution as the 
# B3000 cube 
##############################################################################
sigma_conv_A = np.sqrt(sigma_inst_b_A**2 - sigma_inst_r_A**2)
sigma_conv_px = sigma_conv_A / dlambda_r_A
ndimage.gaussian_filter1d(datacube_r, sigma=sigma_conv_px, axis=0)

##############################################################################
# Spectrally bin the R7000 cube
##############################################################################
# Spectrally interpolate
lambda_vals_comb_A = np.arange(start=lambda_0_b_A, stop=lambda_end_r_A + dlambda_b_A, step=dlambda_b_A)

# datacube_r_interp = np.full((len(lambda_vals_r_interp_A), nrows, ncols), np.nan)
datacube_comb = np.full((len(lambda_vals_comb_A), nrows, ncols), np.nan)
varcube_comb = np.full((len(lambda_vals_comb_A), nrows, ncols), np.nan)

for rr, cc in product(range(nrows), range(ncols)):
    # mask out NaNs
    good_idxs = np.argwhere(~np.isnan(datacube_r[:, rr, cc]))
    # Interpolate
    tck = interpolate.splrep(lambda_vals_r_A[good_idxs], datacube_r[good_idxs, rr, cc])
    datacube_comb[:, rr, cc] = interpolate.splev(x_0_px=lambda_vals_comb_A, tck=tck)

# Repeating for the variance 
# Slightly dodgy, but can't do much better.
for rr, cc in product(range(nrows), range(ncols)):
    # mask out NaNs
    good_idxs = np.argwhere(~np.isnan(varcube_r[:, rr, cc]))
    # Interpolate
    tck = interpolate.splrep(lambda_vals_r_A[good_idxs], varcube_r[good_idxs, rr, cc])
    varcube_comb[:, rr, cc] = interpolate.splev(x_0_px=lambda_vals_comb_A, tck=tck)

fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True)
fig.subplots_adjust(hspace=0.0)
axs[0].plot(lambda_vals_r_A, datacube_r[:, y_0_px, x_0_px], color="k", label="Original data")
axs[0].plot(lambda_vals_comb_A, datacube_comb[:, y_0_px, x_0_px], color="r", label="Interpolated")
axs[1].plot(lambda_vals_r_A, varcube_r[:, y_0_px, x_0_px], color="k", label="Original data")
axs[1].plot(lambda_vals_comb_A, varcube_comb[:, y_0_px, x_0_px], color="r", label="Interpolated")
axs[0].set_xlim([lambda_vals_r_A[0], lambda_vals_r_A[-1]])
axs[0].set_ylim([np.nanmin(datacube_r[:, y_0_px, x_0_px]), np.nanmax(datacube_r[:, y_0_px, x_0_px])])
axs[1].set_ylim([np.nanmin(varcube_r[:, y_0_px, x_0_px]), np.nanmax(varcube_r[:, y_0_px, x_0_px])])
plt.show()

##############################################################################
# Combine the cubes together 
##############################################################################
datacube_comb[:len(lambda_vals_b_A)] = datacube_b
varcube_comb[:len(lambda_vals_b_A)] = varcube_b

# Plot to check...
fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True)
fig.subplots_adjust(hspace=0.0)
axs[0].plot(lambda_vals_r_A / (1 + z), datacube_r[:, y_0_px, x_0_px] + 1e-15, label="R7000 (+ offset)")
axs[0].plot(lambda_vals_b_A / (1 + z), datacube_b[:, y_0_px, x_0_px] + 1e-15, label="B3000 (+ offset)")
axs[0].plot(lambda_vals_comb_A / (1 + z), datacube_comb[:, y_0_px, x_0_px], label="Combined")
axs[0].legend()
axs[1].plot(lambda_vals_r_A / (1 + z), varcube_r[:, y_0_px, x_0_px] + .5e-32, label="R7000 variance (+ offset)")
axs[1].plot(lambda_vals_b_A / (1 + z), varcube_b[:, y_0_px, x_0_px] + .5e-32, label="B3000 variance (+ offset)")
axs[1].plot(lambda_vals_comb_A / (1 + z), varcube_comb[:, y_0_px, x_0_px], label="Combined variance")
axs[1].legend()
plt.show()

##############################################################################
# Save to file
##############################################################################
# Open the individual frame to which the mosaic is aligned
hdu_comb = fits.open(fname_b)

# Overwrite the data with the mosiaced cubes
hdu_comb[0].data = datacube_comb
hdu_comb[1].data = varcube_comb

hdu_comb[0].header["FILEB"] = fname_b
hdu_comb[0].header["FILER"] = fname_r

# Write back to file
hdu_comb.writeto(
    os.path.join(fits_path, "{}_comb.fits".format(obj_name)), output_verify="ignore", overwrite=True)
hdu_comb.close()


