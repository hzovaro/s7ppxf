###############################################################################
#
#   File:       ppxf_binned.py
#   Author:     Henry Zovaro
#   Email:      henry.zovaro@anu.edu.au
#
#   Description:
#   Spatially bin an input S7 datacube using a Voronoi binning scheme,
#   and run ppxf on the binned spectra to determine the star formation history
#   and emission line properties.
#
#   Copyright (C) 2021 Henry Zovaro
#
###############################################################################
from __future__ import print_function
import sys, os

import matplotlib
# matplotlib.use("Agg")

from time import time

from astropy.io import fits
from astropy.wcs import WCS
from astroquery.ned import Ned
import multiprocessing
from scipy import ndimage, constants
import numpy as np
import extinction

from ppxf.ppxf import ppxf
import ppxf.ppxf_util as util
from vorbin.voronoi_2d_binning import voronoi_2d_binning

from cosmocalc import get_dist
from plotting_fns import plot_maps
from log_rebin_errors import log_rebin_errors
from ppxf_plot import ppxf_plot

from IPython.core.debugger import Tracer

##############################################################################
# Plotting settings
##############################################################################
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
matplotlib.rc("font", size=10)
matplotlib.rc("text", usetex=False)
matplotlib.rc("font", **{"family": "serif"})
matplotlib.rc("image", interpolation="nearest")
matplotlib.rc("image", origin="lower")

plt.close("all")
plt.ion()

##############################################################################
# USER OPTIONS
##############################################################################
# Object information
obj_name = sys.argv[1]
grating = "B3000"
check_mask = True  # Whether to pause program execution to check mask used to select spaxels
bad_pixel_ranges_A = []  # Spectral regions to mask out (in Angstroms). format: [[lambda_1, lambda_2], ...]

# Paths
data_dir = "/priv/meggs3/u5708159/S7/" 
ppxf_output_path = os.path.join(data_dir, "ppxf")
fig_path =  os.path.join(ppxf_output_path, "figs") # Where to save figures
output_fits_path =  os.path.join(ppxf_output_path, "fits")  # Path to S7 data cubes
input_fits_path =  os.path.join(data_dir, "0_Cubes")  # Path to S7 data cubes
input_fits_fname = "{}_B.fits".format(obj_name) if grating == "B3000" else "{}_R.fits".format(obj_name)

# Plotting
savefigs = True
plotit = True

# Binning settings
bin_type = "voronoi"
target_SN = 500
im_bin_thresh = 0.05  # Threshold for pixels to be included in the Voronoi binning
mask_radius_px = 25

# ppxf options
ngascomponents = 2  # Number of kinematic components to be fitted to the emission lines
isochrones = "Padova"
auto_adjust_regul = True
mask_NaD = False  # Whether to mask out the Na D doublet - leave False for now

##############################################################################
# For interactive execution
##############################################################################
def hit_key_to_continue():
    key = input("Hit a key to continue or q to quit...")
    if key == 'q':
        sys.exit()
    return

##############################################################################
# For using the FM07 reddening curve in ppxf
##############################################################################
def reddening_fm07(lam, ebv):
    # lam in Angstroms
    # Need to derive A(lambda) from E(B-V)
    # fm07 takes as input lambda and A_V, so we first need to convert E(B-V)
    # into A_V
    A_V = 3.1 * ebv
    A_lambda = extinction.fm07(lam, a_v=A_V, unit='aa')
    fact = 10**(-0.4 * A_lambda)  # Need a minus sign here!
    return fact

##############################################################################
# For printing emission line fluxes
##############################################################################
def sigfig(number, err):
    # Truncate error to one significant figure
    ndigits = int(np.floor(np.log10(err)))
    err_truncated = np.round(err, -ndigits)
    number_truncated = np.round(number, -ndigits)
    return number_truncated, err_truncated, -ndigits

def sci_notation(num, err):
    if num != 0:
        exp = int(np.floor(np.log10(np.abs(num))))
    else:
        exp = 0
    mantissa = num / float(10**exp)
    mantissa_err = err / float(10**exp)
    mantissa_dp, mantissa_err_dp, ndigits = sigfig(mantissa, mantissa_err)
    if exp != 0:
        s = "$ {:." + str(ndigits) + "f} \\pm {:." + str(ndigits) +\
            "f} \\times 10^{{{:d}}} $"
        return s.format(mantissa_dp, mantissa_err_dp, exp)
    else:
        s = "$ {:." + str(ndigits) + "f} \\pm {:." + str(ndigits) + "f}$"
        return s.format(mantissa_dp, mantissa_err_dp)

##############################################################################
# For running in parallel
##############################################################################
def ppxf_helper(args):
    # Parse arguments
    templates, spec_log, spec_err_log, noise_scaling_factor,\
        velscale, start_age_met, good_px, nmoments_age_met, adegree_age_met,\
        mdegree_age_met, dv, lambda_vals_log, regul, reddening_fm07,\
        reg_dim, kinematic_components, gas_component, gas_names,\
        gas_reddening = args

    # Run ppxf
    pp_age_met = ppxf(templates=templates,
          galaxy=spec_log, noise=spec_err_log * noise_scaling_factor,
          velscale=np.squeeze(velscale), start=start_age_met,
          goodpixels=good_px,
          moments=nmoments_age_met, degree=adegree_age_met, mdegree=mdegree_age_met,
          vsyst=dv,
          lam=np.exp(lambda_vals_log),
          regul=regul,
          reddening_func=reddening_fm07,
          reg_dim=reg_dim,
          component=kinematic_components, gas_component=gas_component,
          gas_names=gas_names, gas_reddening=gas_reddening, method="capfit")

    # Return
    return pp_age_met

##############################################################################
# Ojbect information
##############################################################################
# Redshift information
n = Ned.query_object(obj_name)
z = n["Redshift"].data[0]  # Initial estimate of the galaxy redshift
v_sys = n["Velocity"].data[0] # systemic velocity from NED
D_A_Mpc, D_L_Mpc = get_dist(z, H0=70.0, WM=0.3)
kpc_per_as = D_A_Mpc * 1e3 * np.pi / 180.0 / 3600.0
c_km_s = constants.c / 1e3
vel = c_km_s * np.log(1 + z) # Starting guess for systemic velocity (eq.(8) of Cappellari (2017))

# Extinction
t = IrsaDust.get_extinction_table(obj_name)
A_V_Gal = t[2]["A_SandF"]  # S&F2011 - https://irsa.ipac.caltech.edu/cgi-bin/bgTools/nph-bgExec

##############################################################################
# ppxf parameters
##############################################################################
if isochrones == "Padova":
    metals_to_use = ['004', '008', '019']
elif isochrones == "Geneva":
    metals_to_use = ['001', '004', '008', '020', '040']
ssp_template_path = "SSP_templates/SSP{}".format(isochrones)

# pPXF parameters for the age & metallicity + gas fit
adegree_age_met = -1     # Should be zero for age + metallicity fitting
mdegree_age_met = 4     # Should be zero for kinematic fitting
ncomponents = 3    # number of kinematic components. 2 = stars + gas; 3 = stars + 2 * gas
nmoments_age_met = [2 for i in range(ncomponents)]
start_age_met = [[vel, 100.] for i in range(ncomponents)]
fixed_age_met = [[0, 0] for i in range(ncomponents)]
tie_balmer = True if grating == "comb" else False
limit_doublets = False

# pPXF parameters for the stellar kinematics fit
adegree_kin = 12   # Should be zero for age + metallicity fitting
mdegree_kin = 0   # Should be zero for kinematic fitting
nmoments_kin = 2    # 2: only fit radial velocity and velocity dispersion
start_kin = [vel, 100.]
fixed_kin = [0, 0]

# SSP template parameters
# Gonzalez-Delgado spectra_linear have a constant spectral sampling of 0.3 A.
dlambda_A_ssp = 0.30
# Assuming that sigma = dlambda_A_ssp.
FWHM_ssp_A = 2 * np.sqrt(2 * np.log(2)) * dlambda_A_ssp

##############################################################################
# FILE NAMES
##############################################################################
# Description string for filenames
fname_str = "voronoi_{}".format(int(target_SN))

# Figure names
fig_fname = os.path.join(fig_path,
    "{}_ppxf_{}_{}_{}_ngascomponents={}.pdf".format(obj_name, grating, fname_str, isochrones, ncomponents - 1))
fig_regul_fname = os.path.join(fig_path,
    "{}_ppxf_regul_{}_{}_{}_ngascomponents={}.pdf".format(obj_name, grating, fname_str, isochrones, ncomponents - 1))

# Filename of .npy file storing the ppxf instances in case there is a problem 
output_npy_fname = os.path.join(output_fits_path,
    "{}_ppxf_{}_{}_{}_ngascomponents={}.npy".format(obj_name, grating, fname_str, isochrones, ncomponents - 1))

# FITS names
output_fits_fname = os.path.join(output_fits_path,
    "{}_ppxf_{}_{}_{}_ngascomponents={}.fits".format(obj_name, grating, fname_str, isochrones, ncomponents - 1))

if mask_NaD:
    fig_fname = fig_fname.replace("ppxf", "ppxf_mask_NaD")
    fig_regul_fname = fig_regul_fname.replace("ppxf", "ppxf_mask_NaD")
    output_npy_fname = output_npy_fname.replace("ppxf", "ppxf_mask_NaD")
    fits_fname = fits_fname.replace("ppxf", "ppxf_mask_NaD")

print("---------------------------------------------------------------------")
print("FILE NAMES")
print("---------------------------------------------------------------------")
print("Main ppxf figures:\t{}".format(fig_fname))
print("Regul figures:\t\t{}".format(fig_regul_fname))
print(".npy file:\t\t{}".format(output_npy_fname))
print("FITS file:\t\t{}".format(output_fits_fname))
print("---------------------------------------------------------------------")
hit_key_to_continue()

##############################################################################
# Open the data cube containing the galaxy spectra
##############################################################################
# Load FITS file containing WiFeS data
hdu = fits.open(os.path.join(input_fits_path, input_fits_fname))
data_cube = hdu[0].data # TODO: check if we need to crop at all
var_cube = hdu[1].data  # Units of 1e-16 erg/s/cm^2/A

# NaNing out bad spaxels
nan_mask = var_cube >= 1e10
data_cube[nan_mask] = np.nan
var_cube[nan_mask] = np.nan

# NaN out the bottom couple of rows
data_cube[:, :2, :] = np.nan
var_cube[:, :2, :] = np.nan

# Spatial information
N_lambda, nrows, ncols = data_cube.shape
wcs = WCS(hdu[0].header).dropaxis(2)

# Wavelength information
lambda_start_A = hdu[0].header["CRVAL3"]
lambda_end_A = hdu[0].header["CRVAL3"] + \
    hdu[0].header["CDELT3"] * (N_lambda - 1)
dlambda_A = hdu[0].header["CDELT3"]
lambda_vals_linear = np.arange(
    start=lambda_start_A, stop=lambda_end_A + dlambda_A, step=dlambda_A)
lambda_vals_linear = lambda_vals_linear[:data_cube.shape[0]]

# Log-rebin one spectrum to get the logarithmic wavelength scale plus the 
# velscale
_, lambda_vals_log, velscale =\
        util.log_rebin(np.array([lambda_start_A, lambda_end_A]),
                       data_cube[:, 0, 0])

# Instrumental resolution
if grating == "B3000" or grating == "comb":
    FWHM_inst_A = 1.4  # as measured using sky lines in the b3000 grating
elif grating == "R7000":
    FWHM_inst_A = 0.9
sigma_inst_A = FWHM_inst_A / (2 * np.sqrt(2 * np.log(2)))

hdu.close()

##############################################################################
# Correct for galactic extinction
##############################################################################
A_vals = extinction.fm07(lambda_vals_linear, a_v=A_V_Gal)
data_cube = data_cube * 10**(0.4 * A_vals[:, None, None])
var_cube = var_cube * 10**(2 * 0.4 * A_vals[:, None, None])

##############################################################################
# Spatially bin the data cube
##############################################################################
yy, xx = np.meshgrid(range(data_cube.shape[1]), range(
    data_cube.shape[2]), indexing='ij')
yy = yy.flatten()
xx = xx.flatten()

# Create image for binning
if grating == "R7000" or grating == "comb":
    start = np.nanargmin(np.abs(lambda_vals_linear - 5280 * (1 + z)))
    stop = np.nanargmin(np.abs(lambda_vals_linear - 5288 * (1 + z)))
elif grating == "B3000":
    start = np.nanargmin(np.abs(lambda_vals_linear - 4840 * (1 + z)))
    stop = np.nanargmin(np.abs(lambda_vals_linear - 4848 * (1 + z)))

im = np.nanmean(data_cube[start:stop], axis=0)
im_err = np.sqrt(np.nansum(var_cube[start:stop], axis=0)) / (stop - start)


# Creating a mask
# YY, XX = np.meshgrid(range(data_cube.shape[1]), range(
# data_cube.shape[2]), indexing='ij')
# y0, x0 = np.unravel_index(np.nanargmax(im), im.shape)
# px_to_remove = (XX - x0)**2 + (YY - y0)**2 > mask_radius_px**2
# px_to_remove = np.logical_or(px_to_remove, im < im_bin_thresh)

# im[px_to_remove] = np.nan
# im_err[px_to_remove] = np.nan

signal = im.flatten()
noise = im_err.flatten()

# Remove spaxels containing NaNs
good_idxs = [kk for kk in range(len(xx)) if ~np.isnan(
    noise[kk]) and noise[kk] > 0]
xx = xx[good_idxs]
yy = yy[good_idxs]
signal = signal[good_idxs] / 1e-17
noise = noise[good_idxs] / 1e-17

# Carry out the binning
bin_numbers, xNode, yNode, xBar, yBar, sn, nPixels, scale = \
    voronoi_2d_binning(xx, yy, signal, noise, target_SN, plot=1, quiet=0)
plt.show()

# Extract spectra_linear corresponding to each bin
nbins = np.max(bin_numbers) + 1
spectra_linear = np.zeros((nbins, data_cube.shape[0]))
spectra_var = np.zeros((nbins, data_cube.shape[0]))

for ii, n in enumerate(bin_numbers):
    # Zero NaNs before adding
    tmp_d = np.copy(data_cube[:, yy[ii], xx[ii]])
    tmp_v = np.copy(var_cube[:, yy[ii], xx[ii]])
    tmp_d[np.isnan(data_cube[:, yy[ii], xx[ii]])] = 0.0
    tmp_v[np.isnan(data_cube[:, yy[ii], xx[ii]])] = 0.0

    spectra_linear[n] += tmp_d
    spectra_var[n] += tmp_v

spectra_linear_err = np.sqrt(spectra_var)

# Image showing which spaxel belongs to which bin
im_binned = np.full(data_cube.shape[1:], fill_value=np.nan)
for ii, n in enumerate(bin_numbers[:len(yy)]):
    im_binned[yy[ii], xx[ii]] = n

# Calculate the SNR in each bin
SNR_list = np.zeros(nbins)
for ii in range(nbins):
    # Estimate median S/N
    SNR = np.nanmedian(spectra_linear[ii] / spectra_linear_err[ii])
    SNR_list[ii] = SNR
    print("Median SNR in spectrum = {:.4f}".format(SNR))

print("Mean bin SNR = {:.4f}".format(np.nanmedian(SNR_list)))
if check_mask:
    hit_key_to_continue()

##############################################################################
# SSP templates
##############################################################################
# Load the .npz containing the stellar spectra
ssp_template_fnames = ['SSP' + isochrones +
                       '.z' + m + '.npz' for m in metals_to_use]
nmetals = len(ssp_template_fnames)

# All stars_templates_log must have the same number of wavelength values &
# number of age bins!
stars_templates_log = []
stars_templates_linear = []
metallicities = []
for ssp_template_fname in ssp_template_fnames:
    f = np.load(os.path.join(ssp_template_path, ssp_template_fname))
    metallicities.append(f["metallicity"].item())
    ages = f["ages"]
    spectra_ssp_linear = f["L_vals"]
    lambda_vals_ssp_linear = f["lambda_vals_A"]

    # Extract the wavelength range and logarithmically rebin one spectrum
    # to a velocity scale 2x smaller than the stellar template spectra_linear, to
    # determine the size needed for the array which will contain the template
    # spectra.
    spec_ssp_log, lambda_vals_ssp_log, velscale_temp = util.log_rebin(np.array(
        [lambda_vals_ssp_linear[0], lambda_vals_ssp_linear[-1]]),
        spectra_ssp_linear[:, 0], velscale=velscale)
    stars_templates_log.append(np.empty((spec_ssp_log.size, len(ages))))
    stars_templates_linear.append(np.empty((spectra_ssp_linear[:, 0].size,
                                            len(ages))))

    # Quadratic sigma difference in pixels Vazdekis --> SAURON
    # The formula below is rigorously valid if the shapes of the
    # instrumental spectral profiles are well approximated by Gaussians.
    FWHM_diff_A = np.sqrt(FWHM_inst_A**2 - FWHM_ssp_A**2)
    sigma = FWHM_diff_A / (2 * np.sqrt(2 * np.log(2))) / \
        dlambda_A_ssp  # Sigma difference in pixels
    for ii in range(spectra_ssp_linear.shape[1]):
        spec_ssp_linear = spectra_ssp_linear[:, ii]
        spec_ssp_linear = ndimage.gaussian_filter1d(spec_ssp_linear, sigma)
        spec_ssp_log, lambda_vals_ssp_log, velscale_temp =\
            util.log_rebin(np.array(
                [lambda_vals_ssp_linear[0], lambda_vals_ssp_linear[-1]]),
                spec_ssp_linear, velscale=velscale)
        # Normalise templates
        stars_templates_log[-1][:, ii] = spec_ssp_log / np.median(spec_ssp_log)
        stars_templates_linear[-1][:,
                                   ii] = spec_ssp_linear / np.median(spec_ssp_linear)

# String for filename
metal_string = ""
for metal in metallicities:
    metal_string += str(metal).split("0.")[1]
    metal_string += "_"
metal_string = metal_string[:-1]

# Convert to array
stars_templates_log = np.array(stars_templates_log)
stars_templates_log = np.swapaxes(stars_templates_log, 0, 1)
reg_dim = stars_templates_log.shape[1:]
nmetals, nages = reg_dim
stars_templates_log = np.reshape(
    stars_templates_log, (stars_templates_log.shape[0], -1))

# Store the linear spectra_linear too
stars_templates_linear = np.array(stars_templates_linear)
stars_templates_linear = np.swapaxes(stars_templates_linear, 0, 1)
stars_templates_linear = np.reshape(
    stars_templates_linear, (stars_templates_linear.shape[0], -1))

# This line only works if velscale_ratio = 1
dv = (lambda_vals_ssp_log[0] - lambda_vals_log[0]) * c_km_s  # km/s

##############################################################################
# Gas templates
##############################################################################
# Construct a set of Gaussian emission line stars_templates_log.
# Estimate the wavelength fitted range in the rest frame.
gas_templates, gas_names, eline_lambdas = util.emission_lines(
    logLam_temp=lambda_vals_ssp_log,
    lamRange_gal=np.array([lambda_start_A, lambda_end_A]) / (1 + z),
    FWHM_gal=FWHM_inst_A,
    tie_balmer=tie_balmer,
    limit_doublets=limit_doublets,
    vacuum=False
)

##############################################################################
# Merge templates so they can be input to pPXF
##############################################################################
# Combines the stellar and gaseous stars_templates_log into a single array.
# During the PPXF fit they will be assigned a different kinematic
# COMPONENT value
n_ssp_templates = stars_templates_log.shape[1]
# forbidden lines contain "[*]"
n_forbidden_lines = np.sum(["[" in a for a in gas_names])
n_balmer_lines = len(gas_names) - n_forbidden_lines

# Assign component=0 to the stellar templates, component=1 to the Balmer
# gas emission lines templates and component=2 to the forbidden lines.
# if ncomponents == 4:
#     kinematic_components = [0] * n_ssp_templates + \
#         [1] * n_balmer_lines + [2] * n_forbidden_lines + [3] * len(gas_names)
# elif ncomponents == 3:
#     kinematic_components = [0] * n_ssp_templates + \
#         [1] * n_balmer_lines + [2] * n_forbidden_lines

# Here, we lump together the Balmer + forbidden lines into a single kinematic component
if ncomponents == 3:
    kinematic_components = [0] * n_ssp_templates + \
        [1] * len(gas_names) + [2] * len(gas_names)
elif ncomponents == 2:
    kinematic_components = [0] * n_ssp_templates + [1] * len(gas_names)
# gas_component=True for gas templates
gas_component = np.array(kinematic_components) > 0

# If the Balmer lines are tied one should allow for gas reddeining.
# The gas_reddening can be different from the stellar one, if both are fitted.
gas_reddening = 0 if tie_balmer else None

# Combines the stellar and gaseous stars_templates_log into a single array.
# During the PPXF fit they will be assigned a different kinematic
# COMPONENT value
if ncomponents > 2:
    gas_templates = np.concatenate((gas_templates, gas_templates), axis=1)
    gas_names = np.concatenate((gas_names, gas_names))
    eline_lambdas = np.concatenate((eline_lambdas, eline_lambdas))

gas_names_new = []
for ii in range(len(gas_names)):
    gas_names_new.append(gas_names[ii] + " (component {})".format(kinematic_components[ii + n_ssp_templates]))
gas_names = gas_names_new
templates = np.column_stack([stars_templates_log, gas_templates])

##############################################################################
# Loop through spaxels & run ppxf on each
##############################################################################
# Arrays to store results in
met_list = np.zeros((nbins, 1))
age_list = np.zeros((nbins, 1))
stars_vdisp_list = np.zeros((nbins, 1))
stars_vel_list = np.zeros((nbins, 1))
gas_vdisp_list = np.zeros((ncomponents - 1, nbins, 1))
gas_vel_list = np.zeros((ncomponents - 1, nbins, 1))
gas_flux_list = np.zeros((len(gas_names), nbins, 1))
gas_flux_err_list = np.zeros((len(gas_names), nbins, 1))
A_V_list = np.zeros((nbins, 1))
SNR_list = np.zeros((nbins, 1))
norm_list = np.zeros((nbins, 1))
ppxf_age_met_list = [0] * nbins

spectra_log = np.zeros((nbins, N_lambda))
spectra_log_err = np.zeros((nbins, N_lambda))

met_map = np.full((nrows, ncols), fill_value=np.nan)
age_map = np.full((nrows, ncols), fill_value=np.nan)
A_V_map = np.full((nrows, ncols), fill_value=np.nan)
SNR_map = np.full((nrows, ncols), fill_value=np.nan)
gas_flux_maps = np.full((len(gas_names), nrows, ncols), fill_value=np.nan)
gas_flux_err_maps = np.full((len(gas_names), nrows, ncols), fill_value=np.nan)
stars_vdisp_map = np.full((nrows, ncols), fill_value=np.nan)
stars_vel_map = np.full((nrows, ncols), fill_value=np.nan)
gas_vdisp_map = np.full((ncomponents - 1, nrows, ncols), fill_value=np.nan)
gas_vel_map = np.full((ncomponents - 1, nrows, ncols), fill_value=np.nan)

ppxf_bestfit_cube_log = np.full((N_lambda, nrows, ncols), fill_value=np.nan)
binned_spec_cube_log_errs = np.full((N_lambda, nrows, ncols), fill_value=np.nan)
binned_spec_cube_log = np.full((N_lambda, nrows, ncols), fill_value=np.nan)

# Plotting
if plotit:
    # For setting up axes
    fig_nrows = 3
    left = 0.1
    right = 0.1
    middle = 0.1
    bottom = 0.075
    top = 0.075
    cbarax_width = 0.03
    hist_bottom = 0.1

    ax_width = 1 - left - right
    ax_height = (1.0 - hist_bottom - top -
                 (fig_nrows - 1) * middle) / fig_nrows

    ax_kin_width = 2 * ax_width / 3 - (middle / 2)
    ax_age_met_width = 2 * ax_width / 3 - (middle / 2)
    ax_hist_width = 2 * ax_width / 3 - cbarax_width - (middle / 2)

    ax_kin_height = ax_height
    ax_age_met_height = ax_height
    ax_hist_height = ax_height

    ax_kin_left = left
    ax_age_met_left = left
    ax_hist_left = left

    ax_hist_bottom = hist_bottom
    ax_age_met_bottom = ax_hist_bottom + ax_hist_height + middle
    ax_kin_bottom = ax_age_met_bottom + ax_hist_height + middle

    cbarax_left = left + ax_hist_width
    cbarax_bottom = ax_hist_bottom
    cbarax_width = 0.03
    cbarax_height = ax_hist_height

    ax_bin_left = left + ax_age_met_width + middle
    ax_bin_bottom = bottom
    ax_bin_width = ax_width / 3 - (middle / 2)
    ax_bin_height = 1 - bottom - top

    # Axes for plotting the best-fit spectrum
    fig_spec = plt.figure(figsize=(20, 12))
    ax_kin = fig_spec.add_axes([
        ax_kin_left, ax_kin_bottom, ax_kin_width, ax_kin_height])
    ax_age_met = fig_spec.add_axes([
        ax_age_met_left, ax_age_met_bottom, ax_age_met_width, ax_age_met_height])
    ax_hist = fig_spec.add_axes([
        ax_hist_left, ax_hist_bottom, ax_hist_width, ax_hist_height])
    cbarax = fig_spec.add_axes([
        cbarax_left, cbarax_bottom, cbarax_width, cbarax_height])
    ax_bin = fig_spec.add_axes([
        ax_bin_left, ax_bin_bottom, ax_bin_width, ax_bin_height])

    # Open the pdf file
    pp = PdfPages(fig_fname)

    # Figure for auto_adjust_regul
    if auto_adjust_regul:
        fig_regul, ax_regul = plt.subplots()
        pp_regul = PdfPages(fig_regul_fname)

# Save to a numpy .npy file 
pp_arr = np.zeros((nbins, 3), dtype="object")

for ii in range(nbins):
    ##############################################################################
    # Rebin spectra_linear & store in arrays
    ##############################################################################
    spec_linear = spectra_linear[ii]
    spec_err_linear = spectra_linear_err[ii]

    # Estimate median S/N
    SNR = np.nanmedian(spec_linear / spec_err_linear)
    SNR_list[ii] = SNR
    print("Median SNR in spectrum = {:.4f}".format(SNR))

    # Rebin to a log scale
    spec_log, lambda_vals_log, velscale = util.log_rebin(
        np.array([lambda_start_A, lambda_end_A]), spec_linear)

    # Estimate the errors
    spec_err_log = log_rebin_errors(
        spec_linear, spec_err_linear, lambda_start_A, lambda_end_A)

    # Save for later 
    spectra_log[ii] = spec_log
    spectra_log_err[ii] = spec_err_log

    # Mask out regions where the noise vector is zero or inifinite, and where 
    # the spectrum is negative
    bad_px_mask = np.logical_or(spec_err_log <=0, np.isinf(spec_err_log))
    bad_px_mask = np.logical_or(bad_px_mask, spec_log < 0)

    # Mask out manually-defined negative values and problematic regions
    for r_A in bad_pixel_ranges_A:
        r1_A, r2_A = r_A
        r1_px = np.nanargmin(np.abs(np.exp(lambda_vals_log) - r1_A))
        r2_px = np.nanargmin(np.abs(np.exp(lambda_vals_log) - r2_A))
        bad_px_mask[r1_px:r2_px] = True
    good_px = np.squeeze(np.argwhere(~bad_px_mask))

    # Normalize spectrum to avoid numerical issues
    norm = np.median(spec_log[good_px])
    spec_err_log /= norm
    spec_log /= norm  
    spec_err_log[spec_err_log <= 0] = 99999

    ##########################################################################
    # Use pPXF to obtain the stellar age + metallicity, and fit emission lines
    ##########################################################################
    """
    From this paper: https://research-management.mq.edu.au/ws/portalfiles/portal/85573354/85549855.pdf
    "The amount of regularization is controlled by a single regularization parameter. 
    We optimize this parameter for each combined
    spectrum in turn as follows. We first perform a PPXF fit with no
    regularization and scale the errors on our spectra_linear such that chi2 = N,
    where N is the number of good pixels across the spectrum. We
    then choose the regularization parameter such that delta-chi2 = sqrt(2N),
    where delta-chi2 indicates the difference in chi2 values for the regularized 
    and non-regularized fits."
    """

    t = time()
    regul = 0
    delta_chi2_ideal = np.sqrt(2 * len(good_px))
    pp_age_met = ppxf(templates=templates,
                      galaxy=spec_log, noise=spec_err_log,
                      velscale=np.squeeze(velscale), start=start_age_met,
                      goodpixels=good_px,
                      moments=nmoments_age_met, degree=adegree_age_met, mdegree=mdegree_age_met,
                      vsyst=dv,
                      lam=np.exp(lambda_vals_log),
                      regul=regul,
                      reddening_func=reddening_fm07,
                      reg_dim=reg_dim,
                      component=kinematic_components, gas_component=gas_component,
                      gas_names=gas_names, gas_reddening=gas_reddening, method="capfit")
    delta_chi2 = (pp_age_met.chi2 - 1) * len(good_px)
    print("----------------------------------------------------")
    print("Desired Delta Chi^2: %.4g" % delta_chi2_ideal)
    print("Current Delta Chi^2: %.4g" % delta_chi2)
    print("----------------------------------------------------")
    print("Elapsed time in PPXF: %.2f s" % (time() - t))
 
    if not auto_adjust_regul:
        # Manually adjust the regularisation factor.
        print("Scaling noise by {:.4f}...".format(np.sqrt(pp_age_met.chi2)))
        noise_scaling_factor = np.sqrt(pp_age_met.chi2)

        # Manually select the regul parameter value.
        while True:
            key = input("Please enter a value for regul: ")
            if key.isdigit():
                regul = float(key)
                break

        while True:
            t = time()
            pp_age_met = ppxf(templates=templates,
                              galaxy=spec_log, noise=spec_err_log * noise_scaling_factor,
                              velscale=np.squeeze(velscale), start=start_age_met,
                              goodpixels=good_px,
                              moments=nmoments_age_met, degree=adegree_age_met, mdegree=mdegree_age_met,
                              vsyst=dv,
                              lam=np.exp(lambda_vals_log),
                              regul=regul,
                              reddening_func=reddening_fm07,
                              reg_dim=reg_dim,
                              component=kinematic_components, gas_component=gas_component,
                              gas_names=gas_names, gas_reddening=gas_reddening, method="capfit")
            delta_chi2 = (pp_age_met.chi2 - 1) * len(good_px)
            print("----------------------------------------------------")
            print("Desired Delta Chi^2: %.4g" % delta_chi2_ideal)
            print("Current Delta Chi^2: %.4g" % delta_chi2)
            print("----------------------------------------------------")
            print("Elapsed time in PPXF: %.2f s" % (time() - t))

            while True:
                key = input("Enter a new regul value, otherwise press enter: ")
                if key.isdigit() or key == "":
                    break
            if key == "":
                break
            else:
                regul = float(key)

    else:
        # Automatically adjust the regularisation factor.
        print("Scaling noise by {:.4f}...".format(np.sqrt(pp_age_met.chi2)))
        noise_scaling_factor = np.sqrt(pp_age_met.chi2)

        # Run ppxf a number of times & find the value of regul that minimises 
        # the difference between the ideal delta-chi2 and the real delta-chi2.
        regul_vals = np.linspace(0, 2000, 21)
        obj_vals = []  # "objective" fn
        pps = []

        # Input arguments
        args_list = [
            [
                templates, spec_log, spec_err_log, noise_scaling_factor,
                velscale, start_age_met, good_px, nmoments_age_met, adegree_age_met,
                mdegree_age_met, dv, lambda_vals_log, regul, reddening_fm07,
                reg_dim, kinematic_components, gas_component, gas_names,
                gas_reddening
            ] for regul in regul_vals
        ]

        # Run in parallel
        nthreads = min(multiprocessing.cpu_count(), len(args_list))
        print("Running ppxf for bin {} on {} threads...".format(ii, nthreads))
        pool = multiprocessing.Pool(nthreads)
        pps = list(pool.map(ppxf_helper, args_list))
        pool.close()
        pool.join()

        # Determine which is the optimal regul value
        # Quite certain this is correct - see here: https://pypi.org/project/ppxf/#how-to-set-regularization
        regul_vals = [p.regul for p in pps]  # Redefining as pool may not retain the order of the input list
        delta_chi2_vals = [(p.chi2 - 1) * len(good_px) for p in pps]
        obj_vals = [np.abs(delta_chi2 - delta_chi2_ideal) for delta_chi2 in delta_chi2_vals]
        opt_idx = np.nanargmin(obj_vals)

        # If opt_idx is the largest value, then re-run this bin with larger regul values.
        cnt = 2
        while regul_vals[opt_idx] == np.nanmax(regul_vals) and np.nanmax(regul_vals) < 5e3:
            # Input arguments
            regul_vals = np.linspace(np.nanmax(regul_vals), np.nanmax(regul_vals) + 2000, 21)
            args_list = [
                [
                    templates, spec_log, spec_err_log, noise_scaling_factor,
                    velscale, start_age_met, good_px, nmoments_age_met, adegree_age_met,
                    mdegree_age_met, dv, lambda_vals_log, regul, reddening_fm07,
                    reg_dim, kinematic_components, gas_component, gas_names,
                    gas_reddening
                ] for regul in regul_vals
            ]

            # Run in parallel
            print("Re-running ppxf for bin {} on {} threads (iteration {})...".format(ii, nthreads, cnt))
            pool = multiprocessing.Pool(nthreads)
            pps = list(pool.map(ppxf_helper, args_list))
            pool.close()
            pool.join()

            # Determine which is the optimal regul value
            regul_vals = [p.regul for p in pps]  # Redefining as pool may not retain the order of the input list
            delta_chi2_vals = [(p.chi2 - 1) * len(good_px) for p in pps]
            obj_vals = [np.abs(delta_chi2 - delta_chi2_ideal) for delta_chi2 in delta_chi2_vals]
            opt_idx = np.nanargmin(obj_vals)
            cnt += 1

        pp_age_met = pps[opt_idx]

    ##########################################################################
    # Use pPXF to fit the stellar kinematics
    ##########################################################################
    pp_kin = ppxf(templates=stars_templates_log,
                  galaxy=spec_log - pp_age_met.gas_bestfit, noise=spec_err_log * noise_scaling_factor,
                  velscale=np.squeeze(velscale), start=start_kin,
                  goodpixels=good_px,
                  moments=nmoments_kin, degree=adegree_kin, mdegree=mdegree_kin,
                  vsyst=dv,
                  lam=np.exp(lambda_vals_log),
                  method="capfit")
    print("Formal errors:")
    print("     dV    dsigma   dh3      dh4")
    print("".join("%8.2g" % f for f in pp_kin.error * np.sqrt(pp_kin.chi2)))
    print("Elapsed time in pPXF: %.2f s" % (time() - t))

    # Save to Numpy .npy file
    pp_arr[ii, 0] = ii
    pp_arr[ii, 1] = pp_age_met
    pp_arr[ii, 2] = pp_kin
    np.save(output_npy_fname, pp_arr)

    ##########################################################################
    # Reddening
    ##########################################################################
    # Calculate the A_V
    if not tie_balmer and grating == "comb":
        intrinsic_ratios = {
            "Halpha/Hbeta": 2.85,
            "Hgamma/Hbeta": 0.468,
            "Hdelta/Hbeta": 0.259,
        }
        balmer_line_waves = {
            "Hdelta": 4101.734,
            "Hgamma": 4340.464,
            "Hbeta": 4861.325,
            "Halpha": 6562.800,
        }

        for line_1, line_2 in [["Hgamma", "Hbeta"], ["Hdelta", "Hbeta"]]:
            # From p. 384 of D&S
            intrinsic_ratio = intrinsic_ratios[line_1 + "/" + line_2]

            lfmap_1 = pp_age_met.gas_flux[list(gas_names).index(line_1)] * norm
            lfmap_1_err = pp_age_met.gas_flux_error[list(
                gas_names).index(line_1)] * norm

            lfmap_2 = pp_age_met.gas_flux[list(gas_names).index(line_2)] * norm
            lfmap_2_err = pp_age_met.gas_flux_error[list(
                gas_names).index(line_2)] * norm

            ratio = lfmap_1 / lfmap_2
            ratio_err = ratio * ((lfmap_1_err / lfmap_1) **
                                 2 + (lfmap_2_err / lfmap_2)**2)**(0.5)
            ratio_SNR = ratio / ratio_err

            E_ba = 2.5 * (np.log10(ratio)) - 2.5 * np.log10(intrinsic_ratio)
            E_ba_err = 2.5 / np.log(10) * ratio_err / ratio

            # Calculate ( A(Ha) - A(Hb) ) / E(B-V) from extinction curve
            R_V = 3.1
            wave_1_A = np.array([balmer_line_waves[line_1]])
            wave_2_A = np.array([balmer_line_waves[line_2]])

            # A_V is a multiplicative scale factor for the extinction curve.
            # So the below calculation is the same regardless of A_V because
            # we normalise by it.
            E_ba_over_E_BV = float(extinction.fm07(wave_2_A, a_v=1.0) -
                                   extinction.fm07(wave_1_A, a_v=1.0)) /\
                1.0 * R_V

            # Calculate E(B-V)
            E_BV = 1 / E_ba_over_E_BV * E_ba
            E_BV_err = 1 / E_ba_over_E_BV * E_ba_err

            # Calculate A(V)
            A_V = R_V * E_BV
            A_V_err = R_V * E_BV_err

            print(
                "-----------------------------------------------------------------------")
            print("Estimated mean A_V for integrated spectrum using ratio " +
                  line_1 + "/" + line_2 + " (pPXF):")
            print("A_V = {:6.4f} +/- {:6.4f}".format(A_V, A_V_err))
            print(
                "-----------------------------------------------------------------------")
    elif tie_balmer and grating == "comb":
        print("-----------------------------------------------------------------------")
        print("Estimated mean A_V for integrated spectrum using all Balmer lines (calculated by pPXF):")
        print("A_V = {:6.4f}".format(pp_age_met.gas_reddening * 3.1))
        print("-----------------------------------------------------------------------")
    else:
        print("------------------------------------------------------------------------------")
        print("Reddening not calculated due to insufficient Balmer lines in wavelength range")
        print("------------------------------------------------------------------------------")

    ##########################################################################
    # Print emission line fluxes
    ##########################################################################
    print("pPXF emission line fluxes")
    print("-----------------------------------------------------------------------")
    # NOTE: since the input spectrum is in units of erg/s/cm2/A, these fluxes 
    # need to be multilpied by the spectral pixel width in Angstroms to get the 
    # flux in units of erg/s/cm2.
    for name, flux, flux_error in zip(pp_age_met.gas_names,
                                      pp_age_met.gas_flux,
                                      pp_age_met.gas_flux_error):
        try:
            print("{:} \t & {:} \\\\".format(
                name, sci_notation(flux * norm, flux_error * norm)))
        except:
            pass

    ##########################################################################
    # Template weights
    ##########################################################################
    weights_age_met = pp_age_met.weights
    weights_age_met = np.reshape(
        weights_age_met[~gas_component], (nmetals, nages))
    weights_age_met /= np.nansum(weights_age_met)

    weights_kin = pp_kin.weights
    weights_kin = np.reshape(weights_kin, (nmetals, nages))
    weights_kin /= np.nansum(weights_kin)

    ##########################################################################
    # Store in arrays
    ##########################################################################
    # ppxf instance
    norm_list[ii] = norm
    ppxf_age_met_list[ii] = pp_age_met

    # Stellar kinematics
    stars_vel_list[ii] = pp_kin.sol[0]
    stars_vdisp_list[ii] = pp_kin.sol[1]

    # Gas kinematics
    for n in range(1, ncomponents):
        gas_vel_list[n - 1][ii] = pp_age_met.sol[n][0]
        gas_vdisp_list[n - 1][ii] = pp_age_met.sol[n][1]

    # Gas fluxes 
    gas_flux_list[:, ii, 0] = pp_age_met.gas_flux
    gas_flux_err_list[:, ii, 0] = pp_age_met.gas_flux_error

    # Extinction 
    R_V = 3.1
    if pp_age_met.gas_reddening is not None:
        A_V_list[ii] = pp_age_met.gas_reddening * R_V
    else:
        A_V_list[ii] = np.nan

    # Age & metallicity
    met_idx, age_idx = np.unravel_index(
        np.nanargmax(weights_age_met), weights_age_met.shape)
    met_list[ii] = metallicities[met_idx]
    age_list[ii] = ages[age_idx]

    ##########################################################################
    # Plotting the fit
    ##########################################################################
    if plotit:
        # Clear axes
        ax_hist.clear()
        ax_kin.clear()
        ax_age_met.clear()
        ax_bin.clear()
        cbarax.clear()

        # Histogram
        m = ax_hist.imshow(weights_age_met, cmap="magma_r",
                           origin="lower", aspect="auto")
        fig_spec.colorbar(m, cax=cbarax)
        ax_hist.set_yticks(range(len(metallicities)))
        ax_hist.set_yticklabels(["{:.3f}".format(met / 0.02)
                                 for met in metallicities])
        ax_hist.set_ylabel(r"Metallicity ($Z_\odot$)")
        cbarax.set_ylabel("Relative fraction")
        ax_hist.set_xticks(range(len(ages)))
        ax_hist.set_xlabel("Age (Myr)")
        ax_hist.set_title("Best fit stellar population")
        ax_hist.set_xticklabels(["{:}".format(age / 1e6)
                                 for age in ages], rotation="vertical")

        # Kinematic and age & metallicity fits
        ax_age_met.clear()
        ax_kin.clear()
        ppxf_plot(pp_age_met, ax_age_met)
        ppxf_plot(pp_kin, ax_kin)
        ax_age_met.set_title("ppxf fit (age \& metallicity)")
        ax_kin.set_title(r"ppxf fit (kinematics); $v = %.2f$ km s$^{-1}$, $\sigma = %.2f$ km s$^{-1}$" % (
            pp_kin.sol[0] - v_sys, pp_kin.sol[1]))

        # indicate which bin this is
        bin_mask = im_binned == ii
        ax_bin.set_title('Bin {:d} of {:d}, median SNR = {:.2f}'.format(ii, nbins, SNR))
        ax_bin.set_xlabel('X (pixels)')
        ax_bin.set_ylabel('Y (pixels)')
        ax_bin.imshow(im_binned, cmap='tab20')
        ax_bin.imshow(bin_mask, cmap='gray', alpha=0.5)
        fig_spec.canvas.draw()

        # Write to file
        pp.savefig(fig_spec)

        if auto_adjust_regul:
            ax_regul.clear()
            ax_regul.plot(regul_vals, obj_vals, "bo")
            ax_regul.plot(regul_vals[np.nanargmin(obj_vals)], obj_vals[np.nanargmin(obj_vals)], "ro", label="Optimal fit")
            ax_regul.axhline(0, color="gray")
            ax_regul.set_title("Bin {:d} of {:d}".format(ii, nbins))
            ax_regul.set_xlabel("Regularisation parameter")
            ax_regul.set_ylabel(r"$\Delta\chi_{\rm goal}^2 - \Delta\chi^2$")
            ax_regul.legend()
            fig_regul.canvas.draw()
            pp_regul.savefig(fig_regul)

        plt.show()

##########################################################################
# Plot maps of extracted quantities
##########################################################################
if bin_type == "voronoi":
    matplotlib.rc("font", size=16)
    for ii, y, x in zip(bin_numbers, yy, xx):
        # Kinematics
        stars_vel_map[y, x] = stars_vel_list[ii]
        stars_vdisp_map[y, x] = stars_vdisp_list[ii]
        for n in range(1, ncomponents):
            gas_vel_map[n - 1][y, x] = gas_vel_list[n - 1][ii]
            gas_vdisp_map[n - 1][y, x] = gas_vdisp_list[n - 1][ii]

        # Emission line fluxes 
        for l in range(len(gas_names)):
            gas_flux_maps[l][y, x] = gas_flux_list[l][ii]
            gas_flux_err_maps[l][y, x] = gas_flux_err_list[l][ii]
        
        # Extinction
        A_V_map[y, x] = A_V_list[ii]

        # S/N
        SNR_map[y, x] = SNR_list[ii]

        # Age & metallicity
        met_map[y, x] = met_list[ii]
        age_map[y, x] = age_list[ii]

        # Raw binned spectra (log scale)
        binned_spec_cube_log[:, y, x] = spectra_log[ii]
        binned_spec_cube_log_errs[:, y, x] = spectra_log_err[ii]
        ppxf_bestfit_cube_log[:, y, x] = (ppxf_age_met_list[ii].bestfit - ppxf_age_met_list[ii].gas_bestfit) * norm_list[ii]

    # Plot
    if plotit:
        plot_maps([[age_map / 1e6, met_map / 0.20]],
                  [['afmhot', 'magma']],
                  [['Stellar age', 'Stellar metallicity']],
                  [['Myr', r'$Z_\odot$']],
                  [[wcs, wcs]], FIG_WIDTH=12, FIG_HEIGHT=6.5)
        if savefigs:
            pp.savefig(plt.gcf())
        plot_maps([[stars_vel_map, stars_vdisp_map]],
                  [['coolwarm', 'plasma']], [
                  ['Stellar velocity', 'Stellar velocity dispersion']],
                  [[r'km s$^{-1}$', r'km s$^{-1}$']],
                  [[wcs, wcs]], FIG_WIDTH=12, FIG_HEIGHT=6.5)
        if savefigs:
            pp.savefig(plt.gcf())
        for n in range(1, ncomponents):
            plot_maps([[gas_vel_map[n - 1], gas_vdisp_map[n - 1]]],
                      [['coolwarm', 'plasma']],
                      [['Gas velocity (component {})'.format(n), 'Gas velocity dispersion (component {})'.format(n)]],
                      [[r'km s$^{-1}$', r'km s$^{-1}$']],
                      [[wcs, wcs]], FIG_WIDTH=12, FIG_HEIGHT=6.5)
            if savefigs:
                pp.savefig(plt.gcf())

        for ll in range(len(gas_names)):
            plot_maps([[gas_flux_maps[ll], gas_flux_err_maps[ll]]],
                      [['viridis', 'viridis']],
                      [["{} flux".format(gas_names[ll].replace("_", " ")), "{} flux error".format(gas_names[ll]).replace("_", " ")]],
                      [[r"arb. u.", r"arb. u."]],
                      [[wcs, wcs]], FIG_WIDTH=12, FIG_HEIGHT=6.5)
            if savefigs:
                pp.savefig(plt.gcf())

        if grating == "comb":
            plot_maps([[A_V_map]],
                      [['afmhot_r']], [
                      [r'$A_V$']],
                      [['mag']],
                      [[wcs]], FIG_WIDTH=12, FIG_HEIGHT=6.5)
            if savefigs:
                pp.savefig(plt.gcf())

# Close the PDF files
if plotit:
    pp.close()
    if auto_adjust_regul:
        pp_regul.close()

##########################################################################
# Save to FITS file
##########################################################################
hdulist = []
hdulist.append(fits.PrimaryHDU())
hdulist[0].header['NAXIS'] = 3
hdulist[0].header['OBJECT'] = obj_name
hdulist[0].header['FNAME'] = input_fits_fname
hdulist[0].header['ISOCHRN'] = isochrones
hdulist[0].header['NBINS'] = nbins
if bin_type == "voronoi":
    hdulist[0].header['TARGSN'] = target_SN
    hdulist[0].header['SNSTART'] = start
    hdulist[0].header['SNSTOP'] = stop

# Wavelength information
# Because the FITS standard only allows linear axis values, we store the 
# log of the rebinned wavelength values since these will be evenly spaced.
hdulist[0].header['NAXIS3'] = len(lambda_vals_log)
hdulist[0].header['CRPIX3'] = 1
hdulist[0].header['CDELT3'] = lambda_vals_log[1] - lambda_vals_log[0]
hdulist[0].header['CUNIT3'] = 'log Angstroms'
hdulist[0].header['CTYPE3'] = 'Wavelength'
hdulist[0].header['CRVAL3'] = lambda_vals_log[0]

# Also need spatial information in here
hdu = fits.open(os.path.join(input_fits_path, input_fits_fname))
keys = ['NAXIS2', 'CRPIX2', 'CRVAL2', 'CTYPE2',
        'NAXIS1', 'CRPIX1', 'CRVAL1', 'CTYPE1',
        'CD1_1', 'CD1_2',
        'CD2_1', 'CD2_2']
for key in keys:
    hdulist[0].header[key] = hdu[0].header[key]
hdu.close()

# Storing other information
hdulist.append(fits.ImageHDU(data=im_binned, name="Bins"))
hdulist.append(fits.ImageHDU(data=binned_spec_cube_log,
                             name="Binned spectra (log)"))
hdulist.append(fits.ImageHDU(data=binned_spec_cube_log_errs,
                             name="Binned spectra (log) errors"))
hdulist.append(fits.ImageHDU(data=ppxf_bestfit_cube_log,
                             name="Best fit spectra (log)"))
hdulist.append(fits.ImageHDU(data=SNR_map, name="Median SNR"))
hdulist.append(fits.ImageHDU(data=A_V_map, name="A_V"))
for n in range(1, ncomponents):
    hdulist.append(fits.ImageHDU(data=[gas_vel_map[n - 1], gas_vdisp_map[n - 1]],
                                 name="Gas kinematics (component {})".format(n)))
    hdulist[-1].header['BUNIT'] = 'km s^-1'

for l in range(len(gas_names)):
    hdulist.append(fits.ImageHDU(data=[gas_flux_maps[l], gas_flux_err_maps[l]],
                                 name=gas_names[l]))
    hdulist[-1].header['BUNIT'] = 'erg s^-1 cm^-2 dlamba^-1'

hdulist.append(fits.ImageHDU(data=[stars_vel_map, stars_vdisp_map],
                             name="Stellar kinematics"))
hdulist[-1].header['BUNIT'] = 'km s^-1'
hdulist = fits.HDUList(hdulist)

# Save to file
hdulist.writeto(output_fits_fname, overwrite=True)
