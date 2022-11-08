'''
2020-01-10, Dennis Alp, dalp@kth.se

Parses tabular values from scripts (just collects a bunch of numbers).
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
mp = 1.67262192369e-24 # g

# print(dat.dtype.names)

################################################################
obs = ['0781890401',
       '0770380401',
       '0675010401',
       '0149780101',
       '0502020101',
       '0300240501',
       '0604740101',
       '0760380201',
       '0300930301',
       '0765041301',
       '0743650701',
       '0203560201']


xrt = {'0149780101': 'XRT 030206',
       '0203560201': 'XRT 040610',
       '0300240501': 'XRT 060207',
       '0300930301': 'XRT 050925',
       '0502020101': 'XRT 070618',
       '0604740101': 'XRT 100424',
       '0675010401': 'XRT 110621',
       '0743650701': 'XRT 140811',
       '0760380201': 'XRT 151128',
       '0765041301': 'XRT 160220',
       '0770380401': 'XRT 151219',
       '0781890401': 'XRT 161028'}

host = {'XRT 161028': [263.23707,  43.51231],
        'XRT 151219': [173.53037,   0.87409],
        'XRT 110621': [ 37.89582, -60.62918],
        'XRT 030206': [ 29.28776,  37.62768],
        'XRT 070618': None,
        'XRT 060207': None,
        'XRT 100424': [321.79659, -12.03900],
        'XRT 151128': [167.07885,  -5.07495],
        'XRT 050925': [311.43769, -67.64740],
        'XRT 160220': [204.19926, -41.33718],
        'XRT 140811': [ 43.65365,  41.07406],
        'XRT 040610': None}
    

    
for oi in obs:
    # Get coordinates
    ff = open('/Users/silver/box/phd/pro/sne/sbo/src/xmm_' + oi + '.sh')
    for line in ff:
        if line[:7] == 't_rise=':
            t_rise = float(line.split('=')[1])
        elif line[:15] == '# uncertainty (':
            err = float(line.split()[-2].split('=')[-1])
        elif line[:6] == 't_mid=':
            t_mid = float(line.split('=')[1])
        elif line[:7] == 't_fade=':
            t_fade = float(line.split('=')[1])
        elif line[:9] == 'tpeak_min':
            tpeak_min = float(line.split('=')[1])
        elif line[:9] == 'tpeak_max':
            tpeak_max = float(line.split('=')[1])
        elif line[:9] == 'tpeak_bin':
            tpeak_bin = float(line.split('=')[1])
        elif line[:9] == '# DET_ML=':
            ra = float(line.split(',')[1][2:])
            de = float(line.split(',')[2][:-2])
            
    coo = SkyCoord(ra*u.deg, de*u.deg)
    ll = coo.galactic.l.value
    bb = coo.galactic.b.value
    tt = Time(t_rise, scale='tt', format='cxcsec').iso

    # Get N_H
    ff = open('/Users/silver/box/phd/pro/sne/sbo/src/xsp_mc.sh')
    trigger = False
    for line in ff:
        if line[:13] == 'oi=' + oi:
            trigger = True
        elif trigger:
            nh = float(line.strip().split('=')[1])
            break

    # print('\n\n' + oi)
    # print('total duration:', t_fade-t_rise, 's')
    # print('ra, dec:', ra, de)
    # print('cxc:', t_rise, t_mid, t_fade)
    # print('Rise:', tt)
    # print('N_H:', nh*1.e22, 'cm^-2')
    if host[xrt[oi]] is None:
        dist = '\\nodata{}'
    else:
        dist = SkyCoord(*host[xrt[oi]], unit=u.deg).separation(SkyCoord(ra, de, unit=u.deg)).arcsec
        dist = '{0:.1f}'.format(dist)
    print('{5:<30s} &{0:>20.5f} &{1:>20.5f} &{6:>20.5f} &{7:>20.5f} &{4:>19.1f} &{2:>18s} &{3:>24s} \\\\'.format(ra, de, oi, tt, err, dist, ll, bb))
