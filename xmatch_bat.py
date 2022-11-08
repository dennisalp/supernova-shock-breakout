'''
2019-09-26, Dennis Alp, dalp@kth.se

Match XMM observations with known SNe and GRBs.
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
from astropy.io import fits
from astropy import units
from astropy.time import Time

#For LaTeX style font in plots
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rc('text', usetex=True)

# Constants, cgs
cc = 2.99792458e10 # cm s-1
GG = 6.67259e-8 # cm3 g-1 s-2
hh = 6.6260755e-27 # erg s
DD = 51.2 # kpc
pc = 3.086e18 # cm
kpc = 3.086e21 # cm
mpc = 3.086e24 # cm
kev2erg = 1.60218e-9 # erg keV-1
Msun = 1.989e33 # g
Lsun = 3.828e33 # erg s-1
Rsun = 6.957e10 # cm
Tsun = 5772 # K
uu = 1.660539040e-24 # g
SBc = 5.670367e-5 # erg cm-2 K-4 s-1
kB = 1.38064852e-16 # erg K-1





################################################################
print('Loading the XMM Observation Log')
ff = '/Users/silver/box/phd/pro/sne/sbo/xmatch/xmm_obs_log.txt'
with open(ff, 'r') as ff:
    content = ff.readlines()

obs = []
tar = []
xmm = []
tt  = []
exp = []
for ii, line in enumerate(content):
    if line[0] == '#':
        continue
    line = [ll.strip() for ll in line.strip().split('|')]
    if line[6] == '' or float(line[6]) < 1000.:
        continue
    
    obs.append(line[0])
    tar.append(line[1])
    xmm.append(line[2] + ' ' + line[3])
    tt.append(Time(line[4]))
    exp.append(float(line[6]))

obs = np.array(obs)
tar = np.array(tar)
xmm = SkyCoord(xmm, unit=(units.hourangle, units.deg))
tt  = np.array(tt)
exp = np.array(exp)




################################################################
print('Loading the Swift/BAT catalog')
ff = '/Users/silver/box/phd/pro/sne/sbo/xmatch/grb/bat/swift_bat_grb_cat.txt'
with open(ff, 'r') as ff:
    content = ff.readlines()

grb = []
ra = []
de = []
t0 = []
for ii, line in enumerate(content):
    if line[0] == '#':
        continue
    elif 'N/A' in line:
        continue

    line = line.strip().split('|')
    grb.append(line[0].strip())
    ra.append(float(line[4]))
    de.append(float(line[5]))
    tmp = line[3].strip()
    if not tmp[4] == '-':
        tmp = tmp[:4] + '-' + tmp[4:6] + '-' + tmp[6:]
    t0.append(Time(tmp, format='isot', scale='utc'))

grb = np.array(grb)
pos = SkyCoord(ra*units.deg, de*units.deg)



################################################################
print('Crossmatching')
iix, iig, dd, _ = pos.search_around_sky(xmm, 20*units.arcmin)

# All detected GRB positions
ff = open('/Users/silver/Box Sync/phd/pro/sne/sbo/xmatch/grb/bat/detections_bat.txt', 'w')
last = -1
for ii in range(iix.size):
    ix = iix[ii]
    ig = iig[ii]

    if not ig == last:
        buf = ''
        written = False
        buf = buf + 64*'#' + '\n'
        buf = buf + '{0:>11s}, {1:>29s} ({2:s})\n'.format(grb[ig], pos[ig].to_string('hmsdms'), pos[ig].to_string())

    days_before = (t0[ig]-tt[ix]).sec/3600/24

    if days_before < 0.:
        buf = buf + 'Offset: {0:4.1f} arcmin {3:9.2f} days since explosion GRB: {1:<12s} Obs. time: {2:s} Target: {4:s}\n'.format(
            dd[ii].arcmin, grb[ig], tt[ix].iso, -days_before, tar[ix])
        written = True
    if not ii+1 == dd.size and not iix[ii+1] == ix and written:
        buf = buf + 64*'#' + '\n\n\n'
        ff.write(buf)
    last = ig
ff.close()

db()
