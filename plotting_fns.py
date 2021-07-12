###############################################################################
#
# 	File:		plotting_fns.py
#	Author:		Henry Zovaro
#	Email:		henry.zovaro@anu.edu.au
#
#	Description:
#	Utilities for plotting.
#
#	Copyright (C) 2021 Henry Zovaro
#
###############################################################################
from __future__ import division, print_function
import csv
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from astropy.wcs.wcs import WCS

cmap_bad_colour = '#b3b3b3'
FIGSIZE = 8
COLORBAR_FRACTION = 0.046
COLORBAR_PAD = 0.05

###############################################################################

def plot_maps(map_list, cmap_label_list, title_list, unit_list,
              wcs_list=None,
              crop_coords=None,
              mask=None,
              vmin=None, vmax=None,
              FIG_WIDTH=5,
              FIG_HEIGHT=5):

    if type(map_list) != list:
        map_list = [[map_list]]
        cmap_label_list = [[cmap_label_list]]
        title_list = [[title_list]]
        unit_list = [[unit_list]]
    # if type(wcs_list) == WCS:
    #     wcs_list = [[wcs_list]]
    # elif wcs_list is None:
    #     wcs_list = [[None]]

    # Determine the number of rows and cols to plot
    naxis_rows = len(map_list)
    if type(map_list[0]) == list:
        naxis_cols = len(map_list[0])
    else:
        naxis_cols = 1

    # Deal with WCS
    if wcs_list is None:
        wcs_list = [[None] * naxis_cols] * naxis_rows
    elif type(wcs_list) == WCS:
        wcs_list = [[wcs_list] * naxis_cols] * naxis_rows

    # Axis height & width
    left = 0.1
    right = 0.15
    middle = 0.10
    bottom = 0.1
    top = 0.1
    cbarax_width = 0.05 / naxis_cols

    ax_width = (1 - left - (naxis_cols - 1) * middle -
                right - naxis_cols * cbarax_width) / naxis_cols
    ax_height = (1 - bottom - (naxis_rows - 1) * middle - top) / naxis_rows

    # Plot the emission lines and the residuals
    fig = plt.figure(figsize=(naxis_cols * FIG_WIDTH, naxis_rows * FIG_HEIGHT))

    for row in range(naxis_rows):
        # Figure axes
        axs = []
        cmap_list = []
        cbaraxs = []
        for col in range(naxis_cols):
            l = left + col * (cbarax_width + ax_width + middle)
            b = bottom + (naxis_rows - 1 - row) * (ax_height + middle)

            if wcs_list[row][col] is not None:
                axs.append(
                    fig.add_axes([l, b, ax_width, ax_height],
                                 projection=wcs_list[row][col]))
            else:
                axs.append(
                    fig.add_axes([l, b, ax_width, ax_height]))
            cbaraxs.append(
                fig.add_axes([l + ax_width, b, cbarax_width, ax_height]))

            cmap_list.append(matplotlib.cm.get_cmap(cmap_label_list[row][col]))
            cmap_list[-1].set_bad('#b3b3b3')

        # Display map_list in axes
        for map, unit, title, cmap, ax, cbarax, in zip(map_list[row], unit_list[row], title_list[row], cmap_list, axs, cbaraxs):
            # Cropping
            if crop_coords is not None:
                xmin, xmax, ymin, ymax = crop_coords
            else:
                xmin = 0
                ymin = 0
                ymax, xmax = map.shape[:2]

            # Masking
            if mask is not None:
                map[~mask] = np.nan

            if vmin is None:
                vmin = np.nanmin(map[ymin:ymax, xmin:xmax][~np.isinf(map[ymin:ymax, xmin:xmax])])
            if vmax is None:
                vmax = np.nanmax(map[ymin:ymax, xmin:xmax][~np.isinf(map[ymin:ymax, xmin:xmax])])
            m = ax.imshow(map, cmap=cmap, vmin=vmin, vmax=vmax)

            # Apply colourbar
            fig.colorbar(m, cax=cbarax)
            cbarax.set_ylabel(unit)

            ax.axis(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
            ax.set_anchor('SE')
            if wcs_list[row][col] is not None:
                ax.set_ylabel("Dec (J2000)", labelpad=-1.0)
                ax.set_xlabel("RA (J2000)")
            else:
                ax.set_ylabel("Y")
                ax.set_xlabel("X")
            ax.set_title(title)

    fig.show()
    return fig
