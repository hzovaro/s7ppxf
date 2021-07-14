# s7ppxf
Scripts for analysing the stellar populations of galaxies in the S7 sample. The details of the ``ppxf`` implementation can be found in [Zovaro et al. 2020](https://ui.adsabs.harvard.edu/abs/2020MNRAS.499.4940Z/abstract).

The stellar templates provided are those of [Gonzales Delgado](https://www.iaa.csic.es/~rosa/research/synthesis/HRES/ESPS-HRES.html). Both Geneva and Padova isochrones are included.

**Requirements**

Tested in both ``python 2.7`` and ``python 3.6`` using ``IPython``.
The following python packages are required:
- ``ppxf``, used to fit the stellar continuum (https://www-astro.physics.ox.ac.uk/~mxc/software/#ppxf)
- ``vorbin``, used to spatially bin the data cubes (https://www-astro.physics.ox.ac.uk/~mxc/software/#VorBin)
- ``extinction``, used to compute the gas and stellar reddening correction (https://extinction.readthedocs.io/en/latest/)
- ``astropy`` and ``astroquery``
- ``multiprocessing`` for parallel execution
- standard scientific python packages: ``numpy``, ``scipy``, ``matplotlib`` etc.

Before running, you must define an evironment variable ``S7_DIR`` pointing to the top-level directory containing the S7 data, plus other sub-folders which will throw assertion errors if not found:
``export S7_DIR="/path/to/S7/data/"``

**Data**

The script requires the reduced blue and red datacubes from the S7 survey. These are available here: https://miocene.anu.edu.au/S7/

**Other information**

The script ``cosmocalc.py`` contains a modified version of Ned Wright's incredibly useful Javascripy cosmology calculator: http://www.astro.ucla.edu/~wright/CosmoCalc.html 

**Usage**

1. Run ``merge_datacubes.py <object name>`` to merge the blue and red datacubes into a single datacube, saved into a FITS file as "<object name>_COMB.fits". 
2. Run ``ppxf_integrated.py <object name>`` to run ``ppxf`` on a spectrum extracted from the central regions of the combined datacube (or on the red or blue datacubes - this can be configured in the script). The radius of the aperture can be modified in the script.
  3. Run ``ppxf_binned.py <object name>`` to run ``ppxf`` on a spectra extracted from Voronoi bins made from the combined datacube (or on the red or blue datacubes). The minimum continuum S/N of the bins can be modified in the script using the variable ``target_SN`` - larger values produce larger bins, and lower spatial resolution. I recommend a value of 60 - for most object this ensures a median continuum S/N of at least 10 in all bins, which is necessary for ``ppxf`` to produce accurate results. 
