###############################################################################
#
#   File:       save_aperture_spectra.py
#   Author:     Henry Zovaro
#   Email:      henry.zovaro@anu.edu.au
#
#   Description:
#   Extract the spectrum from the central regions of the S7 datacubes and 
#   save to a FITS file.
#
#   Copyright (C) 2021 Henry Zovaro
#
###############################################################################
import sys, os
from tqdm import tqdm
from astropy.io import fits
from astroquery.ned import Ned
from astroquery.irsa_dust import IrsaDust
import numpy as np
import extinction
import pandas as pd

from cosmocalc import get_dist

from IPython.core.debugger import Tracer

##############################################################################
# USER OPTIONS
##############################################################################
grating = "COMB"

##############################################################################
# Paths and filenames
##############################################################################
assert "S7_DIR" in os.environ, 'S7_DIR environment variable is not defined! Make sure it is defined in your .bashrc file: export S7_DIR="/path/to/s7/data/"'
data_dir = os.environ["S7_DIR"]
datacube_path =  os.path.join(data_dir, "0_Cubes")  # Path to S7 data cubes
nucspec_path = os.path.join(data_dir, "3_Nuclear_spectra")
output_path = os.path.join(data_dir, "4_Nuclear_spectra_Re")
for path in [output_path, datacube_path]:
    assert os.path.exists(path), f"Directory {path} does not exist!"

gals = [fname.split("_R.fits")[0] for fname in os.listdir(nucspec_path) if fname.endswith("_R.fits") and not fname.startswith("._")]

##############################################################################
# Object information
##############################################################################
for obj_name in tqdm(gals):

    # Name of input FITS file
    assert grating in ["B3000", "R7000", "COMB"], "grating must be one of B3000, R7000 or COMB!"
    if grating == "COMB":
        datacube_fname = f"{obj_name}_COMB.fits"
        output_fits_fname = f"{obj_name}_COMB_Re.fits"
    elif grating == "B3000":
        datacube_fname = f"{obj_name}_B.fits"
        output_fits_fname = f"{obj_name}_B_Re.fits"
    elif grating == "R7000":
        datacube_fname = f"{obj_name}_R.fits"
        output_fits_fname = f"{obj_name}_R_Re.fits"
    if not os.path.exists(os.path.join(datacube_path, datacube_fname)):
        print(F"WARNING: file {os.path.join(datacube_path, datacube_fname)} not found. Skipping...")
        continue

    # Pre-existing nuclear spectrum FITS file name 
    nucspec_fname = f"{obj_name}_R.fits"

    ##############################################################################
    # Ojbect information
    ##############################################################################
    # Redshift information
    n = Ned.query_object(obj_name)
    try:
        z = n["Redshift"].data[0]  # Initial estimate of the galaxy redshift
    except:
        print(f"WARNING: retrieval of redshift for {gal} failed. Skipping...")
        continue
    D_A_Mpc, D_L_Mpc = get_dist(z, H0=70.0, WM=0.3)

    # load the S7 catalogue
    df_metadata = pd.read_csv(os.path.join(data_dir, "s7_metadata.csv"), comment="$")
    df_metadata = df_metadata.set_index("S7_Name")
    r = df_metadata.loc[obj_name, "HL_Re"]   # radius of aperture in arcsec
    r = 3 if np.isnan(r) else r
    x_0 = df_metadata.loc[obj_name, "S7_nucleus_index_x"]
    y_0 = df_metadata.loc[obj_name, "S7_nucleus_index_y"]

    # Extinction
    t = IrsaDust.get_query_table(obj_name, radius="2deg")
    A_V_Gal = t["ext SandF ref"] * 3.1  # S&F 2011 A_V (= E(B-V) * 3.1)

   ##############################################################################
    # Open the data cube containing the galaxy spectra
    ##############################################################################
    # Load FITS file containing WiFeS data
    hdu = fits.open(os.path.join(datacube_path, datacube_fname))
    data_cube = hdu[0].data # TODO: check if we need to crop at all
    var_cube = hdu[1].data

    N_lambda, nrows, ncols = data_cube.shape

    # Wavelength information
    lambda_start_A = hdu[0].header["CRVAL3"]
    lambda_end_A = hdu[0].header["CRVAL3"] + \
        hdu[0].header["CDELT3"] * (N_lambda - 1)
    dlambda_A = hdu[0].header["CDELT3"]
    lambda_vals_linear = np.arange(
        start=lambda_start_A, stop=lambda_end_A + dlambda_A, step=dlambda_A)
    lambda_vals_linear = lambda_vals_linear[:data_cube.shape[0]]

    hdu.close()

    ##############################################################################
    # Correct for galactic extinction
    ##############################################################################
    A_vals = extinction.fm07(lambda_vals_linear, a_v=A_V_Gal)
    data_cube *= 10**(0.4 * A_vals[:, None, None])
    var_cube *= 10**(2 * 0.4 * A_vals[:, None, None])

    ##############################################################################
    # Spatially bin the data cube
    ##############################################################################
    yy, xx = np.meshgrid(range(data_cube.shape[1]), range(
        data_cube.shape[2]), indexing='ij')

    # Mask out spaxels beyond the given radius
    aperture = (xx - x_0)**2 + (yy - y_0)**2 < r**2
    bin_mask = np.zeros(aperture.shape)
    bin_mask[aperture] = 1
    data_cube[:, ~aperture] = np.nan
    var_cube[:, ~aperture] = np.nan

    # Extract spectrum
    spec_linear = np.nansum(np.nansum(data_cube, axis=1), axis=1) * 4 * np.pi * (D_L_Mpc * 1e6 * 3.086e18)**2  # Needs to be in units of erg/s/A
    spec_err_linear = np.sqrt(np.nansum(np.nansum(var_cube, axis=1), axis=1)) * 4 * np.pi * (D_L_Mpc * 1e6 * 3.086e18)**2  # Needs to be in units of erg/s/A

    # Calculate the SNR
    SNR = np.nanmedian(spec_linear / spec_err_linear)

    ##############################################################################
    # Save to FITS file
    ##############################################################################
    hdulist = fits.open(os.path.join(nucspec_path, nucspec_fname))

    # Replace data in FITS file
    hdulist[0].data = spec_linear
    hdulist[1].data = spec_err_linear
    hdulist[2].data = bin_mask
    
    # Add extra info to header
    hdulist[0].header["CRVAL1"] = lambda_vals_linear[0]
    hdulist[0].header["CDELT1"] = np.diff(lambda_vals_linear)[0]
    hdulist[0].header["NAXIS1"] = N_lambda
    hdulist[0].header["BUNIT"] = ("erg s^-1 A^-1", "Spectrum units")
    hdulist[0].header["APTYPE"] = ("R_e", "Type of aperture")
    hdulist[0].header["MEDSNR"] = (SNR, "Median spectral S/N")
    hdulist[0].header["EXTCORR"] = ("Yes", "Has foreground Galactic extinction been applied?")
    hdulist[0].header["GALAV"] = (float(A_V_Gal), "Foreground Galactic extinction (S&F 2011 A_V)")
    hdulist[0].header["Z"] = (z, "NED redshift")
    hdulist[0].header["DLMPC"] = (D_L_Mpc, "Assumed luminosity distance (Mpc)")
    hdulist[0].header["RE"] = (r, "Assumed effective radius (arcseconds)")
    hdulist[0].header["X0"] = (df_metadata.loc[obj_name, "S7_nucleus_index_x"], "Index of nucleus (arcseconds)")
    hdulist[0].header["Y0"] = (df_metadata.loc[obj_name, "S7_nucleus_index_y"], "Index of nucleus (arcseconds)")

    hdulist.writeto(os.path.join(output_path, output_fits_fname), overwrite=True, output_verify="ignore")

