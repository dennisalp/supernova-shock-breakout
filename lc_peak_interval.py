'''
2019-12-18, Dennis Alp, dalp@kth.se

Simple light curve fitter. Fits linear rise with exponential decay.
'''

from __future__ import division, print_function
import os
from pdb import set_trace as db
import sys
import time
from glob import glob
from datetime import date
from tqdm import tqdm
import subprocess

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.optimize import curve_fit

#For LaTeX style font in plots
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rc('text', usetex=True)



################################################################
oi = sys.argv[1]
ff = glob('/Users/silver/dat/xmm/sbo/{0:s}_repro/{0:s}_ep_*_evt_src_during.fits'.format(oi))
print(ff)
if not len(ff) == 1:
    print(ff)
    gg=wp
ff = ff[0]
dat = fits.open(ff)[1].data
tt = dat['TIME']
ii = tt.argsort()
tt = tt[ii]
cts = np.round(0.25*tt.size).astype('int')
durations = tt[cts:]-tt[:-cts]
ii = np.argmin(durations)
t0 = tt[ii]
t1 = tt[ii+cts]
print('{0:d} counts within {1:f} seconds between {2:f} and {3:f}'.format(cts, t1-t0, t0, t1))

# db()
