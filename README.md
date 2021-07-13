# s7ppxf
Scripts for analysing the stellar populations of galaxies in the S7 sample.

**Requirements**

Tested in both ``python 2.7`` and ``python 3.6`` using ``IPython``.
The following python packages are required:
- ``ppxf``, used to fit the stellar continuum ()
- ``vorbin``, used to spatially bin the data cubes ()
- ``extinction``, used to compute the gas and stellar reddening correction ()
- ``astropy`` and ``astroquery``
- standard scientific python packages: ``numpy``, ``scipy``, ``matplotlib`` etc.

Before running, you must define an evironment variable ``S7_DIR`` pointing to the top-level directory containing the S7 data, plus other sub-folders which will throw assertion errors if not found.

**Data**

The script required the reduced blue and red datacubes from the S7 survey. These are available here: https://miocene.anu.edu.au/S7/

**Other information**

The script ``cosmocalc.py`` contains a modified version of Ned Wright's incredibly useful Javascripy cosmology calculator: http://www.astro.ucla.edu/~wright/CosmoCalc.html 
