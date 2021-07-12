from __future__ import division, print_function
import csv
from IPython.core.debugger import Tracer

import sys

import numpy as np

# SSP templates from here:
# https://www.iaa.csic.es/~rosa/research/synthesis/HRES/ESPS-HRES.html

fname = sys.argv[1]
metallicity = float('0.' + fname.split('.')[-1][1:])

lambda_vals = []
L_vals = []
with open(fname,'r') as f:
	reader = csv.reader(f)
	for line in reader:
		# Remove leading and trailing whitespace 
		line = line[0].lstrip().rstrip()
		# Get the ages of each model
		if line.startswith('# Age'):
			line = line.replace('# Age [yr]  ','')
			ages = line.split()
		# Get the SSP model luminosities at this wavelength
		elif not line.startswith('#'):
			# Get the wavelength value & remove substring from the line
			lambda_val = line.split('  ')[0]
			lambda_vals.append(lambda_val)
			line = line.replace(lambda_val+'  ','')

			# Separate into different ages, so that 
			# age_block[i] = [L[erg/s/A/Mo], Neff/Mo,  Lmin[erg/s/A]]
			L_vals.append(line.split()[0::3])

# Convert to .npz arrays
ages = np.array(ages).astype('float')
L_vals = np.array(L_vals).astype('float')
lambda_vals_A = np.array(lambda_vals).astype('float')

# Save to file for use in ppxf
np.savez(fname + ".npz",ages=ages,lambda_vals_A=lambda_vals_A,L_vals=L_vals,metallicity=metallicity)