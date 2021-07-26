###############################################################################
#
#   File:       ppxf_integrated.py
#   Author:     Henry Zovaro
#   Email:      henry.zovaro@anu.edu.au
#
#   Description:
#   Run ppxf on a spectrum extracted from the central regions of an S7 datacube
#   to determine the star formation history and emission line properties.
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

from ppxf.ppxf import ppxf
import ppxf.ppxf_util as util

from cosmocalc import get_dist
from log_rebin_errors import log_rebin_errors
from ppxf_plot import ppxf_plot

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
# USER OPTIONS
##############################################################################
grating = "R7000"

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
ppxf_output_path = os.path.join(data_dir, "ppxf")
fig_path =  os.path.join(ppxf_output_path, "figs") # Where to save figures
output_fits_path =  os.path.join(ppxf_output_path, "fits")  # Path to S7 data cubes
input_fits_path =  os.path.join(data_dir, "0_Cubes")  # Path to S7 data cubes
for path in [ppxf_output_path, fig_path, output_fits_path, input_fits_path]:
    assert os.path.exists(path), "Directory {} does not exist!".format(path) 

##############################################################################
# load the S7 catalogue
##############################################################################
df_metadata = pd.read_csv(os.path.join(data_dir, "s7_metadata.csv"), comment="$")
df_metadata = df_metadata.set_index("S7_Name")

if len(sys.argv) > 1:
    gals = sys.argv[1:]
else:
    gals = df_metadata.index

fig, ax = plt.subplots(nrows=1, ncols=1)
for obj_name in gals:
    # Name of input FITS file
    assert grating in ["B3000", "R7000", "COMB"], "grating must be one of B3000, R7000 or COMB!"
    if grating == "COMB":
        input_fits_fname = "{}_COMB.fits".format(obj_name)
    elif grating == "B3000":
        input_fits_fname = "{}_B.fits".format(obj_name)
    elif grating == "R7000":
        input_fits_fname = "{}_R.fits".format(obj_name)
    assert os.path.exists(os.path.join(input_fits_path, input_fits_fname)), "File {} does not exist!".format(input_fits_fname)

    ##############################################################################
    # Ojbect information
    ##############################################################################
    r = df_metadata.loc[obj_name, "HL_Re"]   # radius of aperture in arcsec
    x_0 = df_metadata.loc[obj_name, "S7_nucleus_index_x"]
    y_0 = df_metadata.loc[obj_name, "S7_nucleus_index_y"]

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

    hdu.close()

    ##############################################################################
    # Spatially bin the data cube
    ##############################################################################
    yy, xx = np.meshgrid(range(data_cube.shape[1]), range(
        data_cube.shape[2]), indexing='ij')

    # Mask out spaxels beyond the given radius
    aperture = (xx - x_0)**2 + (yy - y_0)**2 < r**2

    im = np.nansum(data_cube, axis=0)

    ax.clear()
    ax.imshow(im)
    ax.set_title(obj_name)
    ax.scatter(x=x_0, y=y_0, c="r", s=50)
    fig.canvas.draw()
    hit_key_to_continue()

