'''
2019-09-03, Dennis Alp, dalp@kth.se

Investigate the EXTraS WP2 catalog of aperiodic variables.
'''

from __future__ import division, print_function
import os
from pdb import set_trace as db
import sys
from glob import glob
from datetime import date
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.wcs import WCS
from astropy.io import fits
from astropy import units

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
# Help functions
def filter_500_lc():
    # Total of 802075
    print('\nLC500\nTotal of', xf.size)
#    g0 = (xf < 3)
#    print('Removed bad 3XMM flags', g0.sum())
#    g0 = g0 & qf
#    print('Removed bad EXTraS flag', g0.sum())
    g0 = (f0 > 0.)
    print('Require positive average flux', g0.sum())
    g0 = g0 & (m0 > 0.)
    print('Require positive average flux', g0.sum())
    g0 = g0 & (p0 < 1-0.999999426696856)
    print('P-value of being constant < 5sigma', g0.sum())
    #g0 = g0 & (nn > 3)
    #print('Require at least 4 bins (100 photons)', g0.sum())
    #g0 = g0 & (sk > 0.)
    #print('Skew > 0, this is actually not supersafe if the sbo covers a large fraction of the exposure', g0.sum())
    return g0

def filter_opt_lc():
    # Total of 802075
    print('\nLCOPT\nTotal of', xf.size)
#    g1 = (xf < 3)
#    print('Removed bad 3XMM flags', g1.sum())
#    g1 = g1 & qf
#    print('Removed bad EXTraS flag', g1.sum())
    g1 = (f1 > 0.)
    print('Require positive average flux', g1.sum())
    g1 = g1 & (m1 > 0.)
    print('Require positive average flux', g1.sum())
    g1 = g1 & (p1 < 1-0.999999426696856)
    print('P-value of being constant < 5sigma', g1.sum())
    return g1

def filter_bb3_lc():
    # Total of 802075
    print('\nBBLC4\nTotal of', xf.size)
#    g2 = (xf < 3)
#    print('Removed bad 3XMM flags', g2.sum())
#    g2 = g2 & qf
#    print('Removed bad EXTraS flag', g2.sum())
    g2 = (f2 > 0.)
    print('Require positive average flux', g2.sum())
    g2 = g2 & (p2 < 1-0.999999426696856)
    print('P-value of being constant < 5sigma', g2.sum())
    return g2



################################################################
# Just read the data
dat = fits.open('/Users/silver/Box Sync/phd/pro/sne/sbo/wp2/extras_aperiodic_light.fits')[1].data
qf = dat['QUALITY_FLAG'] # should be 0, else caution
#print(dat.dtype.names)

dat = fits.open('/Users/silver/Box Sync/phd/pro/sne/sbo/wp2/extras_aperiodic.fits')[1].data
xf = dat['3XMM_SUM_FLAG'] # Clean < 3, else manually flagged
obs = np.array(dat['OBS_ID']).astype('int')
exp = np.array(dat['EXP_ID'])
cam = np.array(dat['CAMERA'])
num = np.array(dat['SRC_NUM'])
src = np.array(dat['SRCID'])
coo = np.c_[dat['RA'],  dat['DEC'], dat['RADEC_ERR']/3600]
co2 = SkyCoord(coo[:,0]*u.deg, coo[:,1]*u.deg, frame='icrs')

f0 = dat['UB_LC500_AVE'] # weighted average of the rate in the uniform optimal bin
e0 = dat['UB_LC500_AVE_ERR']  # uncertainty on the weighted average of the rate in the uniform optimal bin light curve (cts/s)
p0 = dat['UB_LC500_CO_PVAL'] # tail probability for a constant model applied to the uniform optimal bin light curve
n0 = dat['UB_LC500_NBINS'] # number of bins used for the variability analysis of the uniform optimal bin light curve
s0 = dat['UB_LC500_SKEW'] # weighted skewness on the distribution of the rate in the uniform optimal bin light curve
m0 = dat['UB_LC500_MEDIAN'] # median of the distribution of the rate in the uniform optimal bin light curve (cts/s)

f1 = dat['UB_LCOPT_AVE']
e1 = dat['UB_LCOPT_AVE_ERR']
p1 = dat['UB_LCOPT_CO_PVAL']
n1 = dat['UB_LCOPT_NBINS']
s1 = dat['UB_LCOPT_SKEW']
m1 = dat['UB_LCOPT_MEDIAN']

f2 = dat['BB_LC_AVE']
e2 = dat['BB_LC_AVE_ERR']
p2 = dat['BB_LC_CO_PVAL']
n2 = dat['BB_LC_NBLOCKS']
s2 = dat['BB_LC_SKEW']

g0 = filter_500_lc()
g1 = filter_opt_lc()
g2 = filter_bb3_lc()
gg = (g0 | g1 | g2)
print('\nUnion:', (g0 | g1 | g2).sum())
print('Intersection:', (g0 & g1 & g2).sum(), '\n')

# https://www88.lamp.le.ac.uk/extras/WP2/0001730201/WP2products/bblc_0001730201_PN_U002_9.fit
# https://www88.lamp.le.ac.uk/extras/WP2/0001730201/WP2products/ublc_0001730201_PN_U002_9_500s.fit
# https://www88.lamp.le.ac.uk/extras/WP2/0001730201/WP2products/ublc_0001730201_PN_U002_9_opt.fit
# http://www.ledas.ac.uk/arnie5/3xmmdr5_summary.php?id=0001730201_009
h1 = 'wget -O dat/{0:010d}_{1:.2}_{2:.4}_{3:d}_bb3.fits https://www88.lamp.le.ac.uk/extras/WP2/{0:010d}/WP2products/bblc_{0:010d}_{1:.2}_{2:.4}_{3:d}.fit\n'
h2 = 'wget -O dat/{0:010d}_{1:.2}_{2:.4}_{3:d}_500.fits https://www88.lamp.le.ac.uk/extras/WP2/{0:010d}/WP2products/ublc_{0:010d}_{1:.2}_{2:.4}_{3:d}_500s.fit\n'
h3 = 'wget -O dat/{0:010d}_{1:.2}_{2:.4}_{3:d}_opt.fits https://www88.lamp.le.ac.uk/extras/WP2/{0:010d}/WP2products/ublc_{0:010d}_{1:.2}_{2:.4}_{3:d}_opt.fit\n'
h4 = 'wget -O dat/{0:010d}_{1:03d}.html http://www.ledas.ac.uk/arnie5/3xmmdr5_summary.php?id={0:010d}_{1:03d}\n\n'

################################################################
# 
ff = open('/Users/silver/Box Sync/phd/pro/sne/sbo/wp2/dl.sh', 'w')
ff.write('#!/bin/bash\n\nmkdir -p dat\n\n')
for ii, g in enumerate(gg):
    if not g:
        continue
    ff.write(h1.format(obs[ii], cam[ii], exp[ii], num[ii]))
    ff.write(h2.format(obs[ii], cam[ii], exp[ii], num[ii]))
    ff.write(h3.format(obs[ii], cam[ii], exp[ii], num[ii]))

ff.close()
db()
