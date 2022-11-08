'''
2019-09-21, Dennis Alp, dalp@kth.se

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
xmm = SkyCoord(xmm, unit=(units.hourangle, units.deg))
tt  = np.array(tt)
exp = np.array(exp)




################################################################
print('Loading the supernova catalog')
ff = '/Users/silver/box/phd/pro/sne/sbo/xmatch/sne/sn_log.txt'
with open(ff, 'r') as ff:
    content = ff.readlines()

sne = []
dat = []
hos = []
typ = []
nam = []
for ii, line in enumerate(content):
    if line[0] == '#':
        continue
    
    sne.append(line[:32])
    if float(line[32:36]) > 1905.:
        dat.append(Time(float(line[49:62]), format='jd'))
    else:
        dat.append(Time('-'.join(line[32:42].split('/')), format='iso'))
    hos.append(line[81:105].strip())
    typ.append(line[105:117].strip())
    nam.append(line[132:172].strip())

sne = SkyCoord(sne, unit=(units.hourangle, units.deg))
dat = np.array(dat)
hos = np.array(hos)
typ = np.array(typ)
nam = np.array(nam)



################################################################
print('Loading 3XMM DR8')
dr8 = fits.open('/Users/silver/Box Sync/phd/pro/sne/sbo/3xmm_dr8/3XMM_DR8cat_v1.0.fits')[1].data
dr8 = dr8[dr8['MJD_START'].argsort()]
drc = SkyCoord(dr8['RA'], dr8['DEC'], unit=(units.deg, units.deg))
det = dr8['DETID']
src = dr8['SRCID']
num = dr8['SRC_NUM']
ob2 = dr8['OBS_ID']
tt0 = Time(dr8['MJD_START'], format='mjd')
fla = dr8['SUM_FLAG']



################################################################
print('Crossmatching')
iis, iix, d2d, _ = xmm.search_around_sky(sne, 15*units.arcmin)
drs, drx, drd, _ = drc.search_around_sky(sne, 5*units.arcsec)

ff = open('/Users/silver/Box Sync/phd/pro/sne/sbo/xmatch/sne/observed_fields.txt', 'w')
last = -1
for ii in range(iis.size):
    ss = iis[ii]
    xx = iix[ii]
    if not ss == last:
        ff.write(64*'#' + '\n')
        ff.write('Date: {0:<23s} Type: {1:<12s} Name: {2:<40s}\n'.format(dat[ss].iso, typ[ss], nam[ss]))
        ff.write('Coordinates: {0:<29s} Host: {1:<24s} ({2:s})\n\n'.format(sne[ss].to_string('hmsdms'), hos[ss], sne[ss].to_string()))

    ff.write('Epoch: {2:<12s} Offset: {0:4.1f} arcmin Obs. ID: {1:10s} Exposure: {3:3.0f} ks Coordinates: {4:<29s}\n'.format(d2d[ii].arcmin, obs[xx], tt[xx].iso, exp[xx]/1e3, xmm[xx].to_string('hmsdms')))

    if not ii+1 == iis.size and not iis[ii+1] == ss:
        ff.write(64*'#' + '\n\n\n')
    last = ss
ff.close()

ff = open('/Users/silver/Box Sync/phd/pro/sne/sbo/xmatch/sne/detections.txt', 'w')
last = -1
for ii in range(drs.size):
    ss = drs[ii]
    dd = drx[ii]

    if not ss == last:
        buf = ''
        written = False
        buf = buf + 64*'#' + '\n'
        buf = buf + 'Date: {0:<23s} Type: {1:<12s} Name: {2:<40s}\n'.format(dat[ss].iso, typ[ss], nam[ss])
        buf = buf + 'Coordinates: {0:<29s} Host: {1:<24s} ({2:s})\n\n'.format(sne[ss].to_string('hmsdms'), hos[ss], sne[ss].to_string())

    days_before = (dat[ss]-tt0[dd]).sec/3600/24
#    if days_before > 0. and days_before < 100.:
#        buf = buf + 'The following entry is {0:3f} days before explosion.\n'.format(days_before)
#        buf = buf + 'Offset: {0:4.1f} arcsec Epoch: {2:<12s} Obs. ID: {1:10s} Detection: {3:15d} Source: {4:15d} Number: {5:3d} Flag: {6:s}\n'.format(drd[ii].arcsec, ob2[dd], tt0[dd].iso, det[dd], src[dd], num[dd], 'Good' if fla[dd] < 3 else 'Bad')
#        written = True
    if days_before < 0.: # Change this to else to get all
#        buf = buf + 'The following entry is {0:3f} days after explosion.\n'.format(-days_before)
        buf = buf + 'Offset: {0:4.1f} arcsec {7:9.2f} days since explosion Epoch: {2:<12s} Obs. ID: {1:10s} Detection: {3:15d} Source: {4:15d} Number: {5:3d} Flag: {6:s}\n'.format(drd[ii].arcsec, ob2[dd], tt0[dd].iso, det[dd], src[dd], num[dd], 'Good' if fla[dd] < 3 else 'Bad', -days_before)
        written = True
    if not ii+1 == drs.size and not drs[ii+1] == ss and written:
        buf = buf + 64*'#' + '\n\n\n'
        ff.write(buf)
    last = ss
ff.close()

db()
