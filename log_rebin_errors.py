###############################################################################
#
#   File:       log_rebin_errors.py
#   Author:     Henry Zovaro
#   Email:      henry.zovaro@anu.edu.au
#
#   Description:
#   Use a Monte Carlo method to estimate the errors on a spectrum that has 
#   been log-rebinned using ppxf's log_rebin utility.
#
#   Copyright (C) 2021 Henry Zovaro
#
###############################################################################
import numpy as np
import ppxf.ppxf_util as util

def log_rebin_errors(spec,spec_errs,lambda_start_A,lambda_end_A,niters=1000):
    """
        Propagate 1sigma errors corresponding to a spectrum through pPXF's log_rebin
        utility.
        ---
        spec:
            spectrum on a linear wavelength grid 
        spec_errs:
            corresponding 1sigma errors
        lambda_start_A, lambda_end_A:
            start & end wavelengths of the input spectrum
        niters:
            number of random realisations of the input spectra to use to estimate
            the uncertainties on the logarithmically rebinned spectrum.
    """
    specs = []
    for ii in range(niters):
        spec_rand = spec + np.random.normal(loc=0, scale=np.abs(spec_errs))
        spec_log, _, _ = util.log_rebin(np.array([lambda_start_A, lambda_end_A]), spec_rand)
        specs.append(spec_log)
    specs = np.array(specs).T

    # Estimate the errors
    return np.nanstd(specs,axis=1)
