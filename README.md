# s7ppxf
Scripts for analysing the stellar populations of galaxies in the S7 sample. The details of the ``ppxf`` implementation can be found in [Zovaro et al. 2020](https://ui.adsabs.harvard.edu/abs/2020MNRAS.499.4940Z/abstract)

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

Before running, you must define an evironment variable ``S7_DIR`` pointing to the top-level directory containing the S7 data, plus other sub-folders which will throw assertion errors if not found.

**Data**

The script required the reduced blue and red datacubes from the S7 survey. These are available here: https://miocene.anu.edu.au/S7/

**Other information**

The script ``cosmocalc.py`` contains a modified version of Ned Wright's incredibly useful Javascripy cosmology calculator: http://www.astro.ucla.edu/~wright/CosmoCalc.html 
