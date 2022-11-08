'''
2020-01-13, Dennis Alp, dalp@kth.se

Compute the parameter distributions from the XSPEC MC output.
'''

from __future__ import division, print_function
import os
from pdb import set_trace as db
import sys
from glob import glob
import time
from datetime import date
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy import units as u
from astropy.io import fits
from astropy import units
from scipy.optimize import curve_fit

def fun(xx, c1, aa, bb, g1, g2, uu, vv):
    print(c1, aa, bb, g1, g2, uu, vv)
    return (1-c1*np.exp(aa*xx**g1))*uu**vv*np.exp(bb*xx**g2)

xx=np.linspace(0.03, 3., 10000)
tot = 8.3783e-08
dat = np.array([
    [0.03, 8.0013e-10],
    [0.04, 4.5795e-09],
    [0.05, 1.1716e-08],
    [0.06, 2.0566e-08],
    [0.10, 5.0855e-08],
    [0.20, 7.5842e-08],
    [0.30, 8.0882e-08],
    [0.60, 8.3335e-08],
    [1.00, 8.2879e-08],
    [1.30, 7.9702e-08],
    [2.00, 6.3203e-08],
    [3.00, 3.8925e-08]])
tt = dat[:,0]
ff = dat[:,1]

guess = np.array([5., -5, -0.01, 0.3, 1., 1., 1.])
pars, covar = curve_fit(fun, tt, ff/tot, guess, maxfev=99999, sigma=ff/tot)
#coef = np.polyfit(tt, ff/tot, 4)
plt.loglog(tt, ff/tot)
plt.loglog(xx, fun(xx, *pars))
#plt.semilogx(xx, np.polyval(coef, xx))
plt.show()
db()

# energies 0.0000001 10000. 100000 log

# ========================================================================
# Model bbody<1> Source No.: 1   Active/Off
# Model Model Component  Parameter  Unit     Value
#  par  comp
#    1    1   bbody      kT         keV      0.100000     +/-  0.0          
#    2    1   bbody      norm                1.00000      +/-  0.0          
# ________________________________________________________________________

# XSPEC12>flux 0.3 10.
#  Model Flux    69.441 photons (5.0855e-08 ergs/cm^2/s) range (0.30000 - 10.000 keV)
# XSPEC12>flux 0.0000001 10000.
#  Model Flux    193.59 photons (8.3783e-08 ergs/cm^2/s) range (1.0000e-07 - 10000. keV)


# ========================================================================
# Model bbody<1> Source No.: 1   Active/Off
# Model Model Component  Parameter  Unit     Value
#  par  comp
#    1    1   bbody      kT         keV      1.00000      +/-  0.0          
#    2    1   bbody      norm                1.00000      +/-  0.0          
# ________________________________________________________________________

# XSPEC12>flux 0.3 10.
#  Model Flux    18.987 photons (8.2879e-08 ergs/cm^2/s) range (0.30000 - 10.000 keV)
# XSPEC12>flux 0.0000001 10000.
#  Model Flux    19.359 photons (8.3783e-08 ergs/cm^2/s) range (1.0000e-07 - 10000. keV)


# ========================================================================
# Model bbody<1> Source No.: 1   Active/Off
# Model Model Component  Parameter  Unit     Value
#  par  comp
#    1    1   bbody      kT         keV      0.300000     +/-  0.0          
#    2    1   bbody      norm                1.00000      +/-  0.0          
# ________________________________________________________________________

# XSPEC12>flux 0.3 10.
#  Model Flux     55.03 photons (8.0882e-08 ergs/cm^2/s) range (0.30000 - 10.000 keV)
# XSPEC12>flux 0.0000001 10000.
#  Model Flux     64.53 photons (8.3783e-08 ergs/cm^2/s) range (1.0000e-07 - 10000. keV)
