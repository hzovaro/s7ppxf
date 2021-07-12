#!/usr/bin/env python
###############################################################################
#
# A handy script for computing cosmological quantities.
#
# Adapted from Ned Wright's Javascript cosmology calculator,
# available http://www.astro.ucla.edu/%7Ewright/CosmoCalc.html
#
###############################################################################
import numpy as np
from scipy import constants

###############################################################################
def v_doppler_to_v_pec(v, z_cos, units):
    """
    Convert Doppler velocities, derived using

        1 + z_obs = sqrt((1 + v_Doppler/c) / (1 - v_Doppler/c))

    to peculiar velocities using eqn. 10 from Capperllari & Emsellem 2017

        1 + z_obs = (1 + z_cos) * (1 + v_pec/c)
    """
    assert units == "km/s" or units == "m/s", "units must be km/s or m/s!"
    c_m_s = constants.c  # equal to 299792.458 * 1e3 (checked)
    c_km_s = constants.c / 1e3
    if units == "m/s":
        z_obs = np.sqrt((1 + v / c_m_s) / (1 - v / c_km_s)) - 1
        v_pec = c_m_s * ((z_obs - z_cos) / (1 + z_cos))
    elif units == "km/s":
        z_obs = np.sqrt((1 + v / c_km_s) / (1 - v / c_km_s)) - 1
        v_pec = c_km_s * ((z_obs - z_cos) / (1 + z_cos))
    return v_pec

###############################################################################
def v_ppxf_to_v_pec(v, z_cos, units):
    """
    Convert velocities computed by ppxf, derived using eqn. 8 from Capperllari & Emsellem 2017

        v = c * ln(1 + z)

    to peculiar velocities using eqn. 10 from Capperllari & Emsellem 2017

        1 + z_obs = (1 + z_cos) * (1 + v_pec/c)    

    NOTE: only valid if the input spectra were NOT corrected for redshift
    before being entered to ppxf!
    """
    assert units == "km/s" or units == "m/s", "units must be km/s or m/s!"
    c_m_s = constants.c  # equal to 299792.458 * 1e3 (checked)
    c_km_s = constants.c / 1e3
    if units == "m/s":
        z_obs = np.exp(v / c_m_s) - 1
        v_pec = c_m_s * ((z_obs - z_cos) / (1 + z_cos))
    elif units == "km/s":
        z_obs = np.exp(v / c_km_s) - 1
        v_pec = c_km_s * ((z_obs - z_cos) / (1 + z_cos))
    return v_pec


###############################################################################
def get_dist(z,
             H0=70.0,
             WM=0.30,
             WV=0.70,
             printit=False):

    if not WV:
        WV = 1.0 - WM - 0.4165 / (H0 * H0)    # Omega(vacuum) or lambda

# initialize constants
    WR = 0.                # Omega(radiation)
    WK = 0.                # Omega curvaturve = 1-Omega(total)
    c = 299792.458    # velocity of light in km/sec
    Tyr = 977.8        # coefficent for converting 1/H into Gyr
    DTT = 0.5            # time from z to now in units of 1/H0
    DTT_Gyr = 0.0    # value of DTT in Gyr
    age = 0.5            # age of Universe in units of 1/H0
    age_Gyr = 0.0    # value of age in Gyr
    zage = 0.1         # age of Universe at redshift z in units of 1/H0
    zage_Gyr = 0.0    # value of zage in Gyr
    DCMR = 0.0         # comoving radial distance in units of c/H0
    DCMR_Mpc = 0.0
    DCMR_Gyr = 0.0
    DA = 0.0             # angular size distance
    DA_Mpc = 0.0
    DA_Gyr = 0.0
    kpc_DA = 0.0
    DL = 0.0             # luminosity distance
    DL_Mpc = 0.0
    DL_Gyr = 0.0     # DL in units of billions of light years
    V_Gpc = 0.0
    a = 1.0                # 1/(1+z), the scale factor of the Universe
    az = 0.5             # 1/(1+z(object))

    h = H0 / 100.
    # includes 3 massless neutrino species, T0 = 2.72528
    WR = 4.165E-5 / (h * h)
    WK = 1 - WM - WR - WV
    az = 1.0 / (1 + 1.0 * z)
    age = 0.
    n = 1000                 # number of points in integrals
    for i in range(n):
        a = az * (i + 0.5) / n
        adot = np.sqrt(WK + (WM / a) + (WR / (a * a)) + (WV * a * a))
        age = age + 1. / adot

    zage = az * age / n
    zage_Gyr = (Tyr / H0) * zage
    DTT = 0.0
    DCMR = 0.0

# do integral over a=1/(1+z) from az to 1 in n steps, midpoint rule

    for i in range(n):
        a = az + (1 - az) * (i + 0.5) / n
        adot = np.sqrt(WK + (WM / a) + (WR / (a * a)) + (WV * a * a))
        DTT = DTT + 1. / adot
        DCMR = DCMR + 1. / (a * adot)

    DTT = (1. - az) * DTT / n
    DCMR = (1. - az) * DCMR / n
    age = DTT + zage
    age_Gyr = age * (Tyr / H0)
    DTT_Gyr = (Tyr / H0) * DTT
    DCMR_Gyr = (Tyr / H0) * DCMR
    DCMR_Mpc = (c / H0) * DCMR

# tangential comoving distance

    ratio = 1.00
    x = np.sqrt(np.abs(WK)) * DCMR
    if x > 0.1:
        if WK > 0:
            ratio = 0.5 * (np.exp(x) - np.exp(-x)) / x
        else:
            ratio = np.sin(x) / x
    else:
        y = x * x
        if WK < 0:
            y = -y
        ratio = 1. + y / 6. + y * y / 120.
    DCMT = ratio * DCMR
    DA = az * DCMT
    DA_Mpc = (c / H0) * DA
    kpc_DA = DA_Mpc / 206.264806
    DA_Gyr = (Tyr / H0) * DA
    DL = DA / (az * az)
    DL_Mpc = (c / H0) * DL
    DL_Gyr = (Tyr / H0) * DL

# comoving volume computation

    ratio = 1.00
    x = np.sqrt(np.abs(WK)) * DCMR
    if x > 0.1:
        if WK > 0:
            ratio = (0.125 * (np.exp(2. * x) - np.exp(-2. * x)) -
                     x / 2.) / (x * x * x / 3.)
        else:
            ratio = (x / 2. - np.sin(2. * x) / 4.) / (x * x * x / 3.)
    else:
        y = x * x
        if WK < 0:
            y = -y
        ratio = 1. + y / 5. + (2. / 105.) * y * y
    VCM = ratio * DCMR * DCMR * DCMR / 3.
    V_Gpc = 4. * np.pi * ((0.001 * c / H0)**3) * VCM

    if printit:
        print('For H_o = ' + '%1.1f' % H0 + ', Omega_M = ' + '%1.2f' % WM +
                ', Omega_vac = ')
        print('%1.2f' % WV + ', z = ' + '%1.3f' % z)
        print('It is now ' + '%1.1f' % age_Gyr + ' Gyr since the Big Bang.')
        print('The age at redshift z was ' + '%1.1f' % zage_Gyr + ' Gyr.')
        print('The light travel time was ' + '%1.1f' % DTT_Gyr + ' Gyr.')
        print('The comoving radial distance, which goes into Hubbles law, is')
        print('%1.1f' % DCMR_Mpc + ' Mpc or ' + '%1.1f' % DCMR_Gyr + ' Gly.')
        print('The comoving volume within redshift z is ' + '%1.1f' %
              V_Gpc + ' Gpc^3.')
        print('The angular size distance D_A is ' + '%1.1f' %
              DA_Mpc + ' Mpc or')
        print('%1.1f' % DA_Gyr + ' Gly.')
        print('This gives a scale of ' + '%.2f' % kpc_DA + ' kpc/".')
        print('The luminosity distance D_L is ' + '%1.1f' %
              DL_Mpc + ' Mpc or ' + '%1.1f' % DL_Gyr + ' Gly.')
        print('The distance modulus, m-M, is ' + '%1.2f' %
              (5 * np.log10(DL_Mpc * 1e6) - 5))

    return DA_Mpc, DL_Mpc


def get_redshift(kpc_per_as_target, tol=0.0001, H0=70.0, WM=0.30, WV=0.70):
    """
    Given a desired physical scale and cosmological parameters, determine the 
    corresponding redshift.
    """
    # First, check that the input physical scale is possible under the given
    # cosmology.
    kpc_per_as_max = 0.1
    kpc_per_as_max_old = 10
    z_vals = np.linspace(0.01, 3, 100)
    while np.abs(kpc_per_as_max - kpc_per_as_max_old) / kpc_per_as_max > tol:
        kpc_per_as_max_old = kpc_per_as_max
        kpc_per_as_vals = np.zeros(z_vals.shape)
        for ii in range(len(z_vals)):
            D_A_Mpc, _ = get_dist(z_vals[ii], H0=H0, WM=WM, WV=WV)
            kpc_per_as_vals[ii] = D_A_Mpc * 1e3 * 1 * np.pi / 180 / 3600
        idx_max = np.nanargmax(kpc_per_as_vals)
        kpc_per_as_max = kpc_per_as_vals[idx_max]
        z_max = z_vals[idx_max]
        z_vals = np.linspace(z_vals[idx_max - 1], z_vals[idx_max + 1], 100)
    print("get_redshift(): For a H0={:.2f} km/s/Mpc, WM={:.2f}, WV={:.2f} cosmology, the maximum physical scale is {:.6f} kpc/arcsec which occurs at z = {:.6f}.".format(H0, WM, WV, kpc_per_as_max, z_max))

    assert kpc_per_as_target < kpc_per_as_max,\
        "get_redshift(): Target physical scale is greater than is permitted by the input cosmology!"

    # Iterate to find the redshift that gives the desired physical scale.
    # Upper branch
    z_vals = np.linspace(z_max, 100, 100)
    niters = 0
    kpc_per_as_best = 0
    while np.abs(kpc_per_as_best - kpc_per_as_target) / kpc_per_as_target > tol and niters < 100:
        niters += 1
        kpc_per_as_vals = np.zeros(z_vals.shape)
        for ii in range(len(z_vals)):
            D_A_Mpc, _ = get_dist(z_vals[ii], H0=H0, WM=WM, WV=WV)
            kpc_per_as_vals[ii] = D_A_Mpc * 1e3 * 1 * np.pi / 180 / 3600
        idx_best = np.nanargmin(np.abs(kpc_per_as_vals - kpc_per_as_target))
        kpc_per_as_best = kpc_per_as_vals[idx_best]
        z_best_upper = z_vals[idx_best]
        # Tracer()()
        z_vals = np.linspace(z_vals[idx_best - 1], z_vals[idx_best + 1], 100)
    print("get_redshift(): z = {:.6f}, corresponding to {:.6f} kpc/arcsec (target = {:.6f} kpc/arcsec)".format(z_best_upper, kpc_per_as_best, kpc_per_as_target))

    # Lower branch
    z_vals = np.linspace(0.0001, z_max, 100)
    niters = 0
    kpc_per_as_best = 0
    while np.abs(kpc_per_as_best - kpc_per_as_target) / kpc_per_as_target > tol and niters < 100:
        niters += 1
        kpc_per_as_vals = np.zeros(z_vals.shape)
        for ii in range(len(z_vals)):
            D_A_Mpc, _ = get_dist(z_vals[ii], H0=H0, WM=WM, WV=WV)
            kpc_per_as_vals[ii] = D_A_Mpc * 1e3 * 1 * np.pi / 180 / 3600
        idx_best = np.nanargmin(np.abs(kpc_per_as_vals - kpc_per_as_target))
        kpc_per_as_best = kpc_per_as_vals[idx_best]
        z_best_lower = z_vals[idx_best]
        z_vals = np.linspace(z_vals[idx_best - 1], z_vals[idx_best + 1], 100)
    print("get_redshift(): z = {:.6f}, corresponding to {:.6f} kpc/arcsec (target = {:.6f} kpc/arcsec)".format(z_best_lower, kpc_per_as_best, kpc_per_as_target))

    return z_best_upper, z_best_lower
