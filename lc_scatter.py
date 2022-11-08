'''
2019-12-18, Dennis Alp, dalp@kth.se

Scatter plot of photons, energy vs time.
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
cwd = '/Users/silver/dat/xmm/sbo/' + oi + '_repro'
os.chdir(cwd)

ff = open('/Users/silver/box/phd/pro/sne/sbo/src/xmm_' + oi + '.sh')
for line in ff:
    if line[:7] == 't_rise=':
        t_rise = float(line.split('=')[1])
    elif line[:6] == 't_mid=':
        t_mid = float(line.split('=')[1])
    elif line[:7] == 't_fade=':
        t_fade = float(line.split('=')[1])

ff = oi + '_ep_clean_evt_src_during.fits'
dat = fits.open(ff)[1].data
tt = dat['TIME']
pi = dat['PI']/1.e3

plt.semilogy(tt-t_rise, pi, '.k')
plt.title(oi)
plt.xlabel('Time (s)')
plt.ylabel('Energy (keV)')
plt.ylim([0.1, 10.])
plt.show()
db()
