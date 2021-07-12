#!/usr/bin/env python
from __future__ import print_function

import glob
from os import path, listdir
from time import clock

from astropy.io import fits
from scipy import ndimage
import numpy as np

from miscutils import printutils


import matplotlib.pyplot as plt

from IPython.core.debugger import Tracer

################################################################################
# Prepare the galaxy spectrum
################################################################################
# Spaxel to plot
x = 20
y = 18
sz = 4

# Load FITS file containing WiFeS data
datacube_fname = '/Users/azovaro/Documents/GPS & CSS sources/Scripts/WiFeS/fits/mo/b3000/mosaic_pcube_skysub_strict.fits'
hdu = fits.open(datacube_fname)
spectrum_gal_linear = np.nansum(np.nansum(hdu[0].data[:,y-sz//2:y+sz//2,x-sz//2:x+sz//2],axis=1),axis=1)
spectrum_gal_var_linear = np.nansum(np.nansum(hdu[1].data[:,y-sz//2:y+sz//2,x-sz//2:x+sz//2],axis=1),axis=1)
spectrum_gal_err_linear = np.sqrt(spectrum_gal_var_linear)

# Wavelength information
lambda_start_gal_A = hdu[0].header['CRVAL3']
lambda_end_gal_A = hdu[0].header['CRVAL3'] + hdu[0].header['CDELT3']*(hdu[0].header['NAXIS3'] - 1)
dlambda_A_gal = hdu[0].header['CDELT3']
lambda_vals_gal_linear = np.arange(start=lambda_start_gal_A,stop=lambda_end_gal_A+dlambda_A_gal,step=dlambda_A_gal)
FWHM_gal = 2.3 * dlambda_A_gal # Resolution is about 2.3 pixels according to the manual (need to check this!)

# Redshift of source
z = 0.0189 # Initial estimate of the spectrum_gal_log redshift

################################################################################
# Prepare the SSP templates
################################################################################
# Parameters
dlambda_A_ssp = 0.30    # Gonzalez-Delgado spectra have a constant spectral sampling of 0.3 A.
FWHM_ssp = 2 * np.sqrt(2 * np.log(2)) * dlambda_A_ssp         # Assuming that sigma = dlambda_A_ssp.
velscale_ratio = 2      # adopts 2x higher spectral sampling for templates than for spectrum_gal_log

# Load the .npz containing the stellar spectra
ssp_template_path = '/Users/azovaro/Documents/GPS & CSS sources/Scripts/SSP_templates/SSPGeneva/'
ssp_template_fnames = [f for f in listdir(ssp_template_path) if f.endswith('.npz')]
# ssp_template_fnames = ['SSPPadova.z004.npz']
nmetals = len(ssp_template_fnames)

# All templates must have the same number of wavelength values & number of age bins!
templates = []
for ssp_template_fname in ssp_template_fnames:
    f = np.load(ssp_template_path + ssp_template_fname)
    ages = f['ages']
    spectra_ssp_linear = f['L_vals']
    lambda_vals_ssp_linear = f['lambda_vals_A']
    templates.append(np.empty((spectra_ssp_linear.shape[0], len(ages))))

    # Quadratic sigma difference in pixels Vazdekis --> SAURON
    # The formula below is rigorously valid if the shapes of the
    # instrumental spectral profiles are well approximated by Gaussians.
    FWHM_diff = np.sqrt(FWHM_gal**2 - FWHM_ssp**2)
    sigma = FWHM_diff / (2 * np.sqrt(2 * np.log(2))) / dlambda_A_ssp  # Sigma difference in pixels
    for ii in range(spectra_ssp_linear.shape[1]):        
        spectrum_ssp_linear = spectra_ssp_linear[:,ii]
        spectrum_ssp_linear = ndimage.gaussian_filter1d(spectrum_ssp_linear, sigma)
        templates[-1][:, ii] = spectrum_ssp_linear/np.median(spectrum_ssp_linear)

# Convert to array 
templates = np.array(templates)
templates = np.swapaxes(templates, 0, 1)
nmetals, nages = templates.shape[1:]

plt.ion()
fig, ax = plt.subplots(nrows=1,ncols=1)
for ix, age in enumerate(ages):
    ax.clear()
    ax.set_title(r'$t = $' + '{:.2f}'.format(age / 1e6) + r' Myr')
    # ax.set_title(r'$Z = $' + '{:.2f}'.format(metallicity) + r' $Z_{\odot}$')
    ax.plot(lambda_vals_gal_linear/(1+z),spectrum_gal_linear/np.median(spectrum_gal_linear),'gray',linewidth=1.5)
    ax.plot(lambda_vals_ssp_linear, templates[:,:,ix],)
    ax.set_xlim([min(lambda_vals_gal_linear/(1+z)),max(lambda_vals_gal_linear/(1+z))])
    ax.set_ylim([0.9*min(templates[:,:,ix].flatten()),1.1*max(templates[:,:,ix].flatten())])
    fig.canvas.draw()
    plt.show()
    printutils.hit_key_to_continue()
plt.ioff()