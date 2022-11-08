'''
2019-09-27, Dennis Alp, dalp@kth.se

Investigate the EXTraS WP4 catalog of new transients.
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
# Just read the data
gai = fits.open('/Users/silver/Box Sync/phd/pro/sne/sbo/wp4/gaia.fits')[1].data
dat = fits.open('/Users/silver/Box Sync/phd/pro/sne/sbo/wp4/extras_transients.fits')[1].data
print(dat.dtype.names)

obs = np.array(dat['OBS_ID']).astype('int')
coo = np.c_[dat['RA'],  dat['DEC'], dat['RADEC_ERR']/3600]
co2 = SkyCoord(coo[:,0]*u.deg, coo[:,1]*u.deg, frame='icrs')
gal = np.c_[dat['LII'], dat['BII'], dat['RADEC_ERR']/3600]
cts = np.c_[dat['EP_0_SCTS'].astype('float'), dat['EP_0_SCTS_ERR'].astype('float')]
tt = np.c_[dat['T_START'], dat['T_STOP']]



################################################################
# Print a formated file for use with SIMBAD, Gaia, and NED
# This creates a bash script that queries NED and has to be run separetely
# http://simbad.u-strasbg.fr/simbad/sim-fcoo
# https://ned.ipac.caltech.edu/Documents/Guides/Interface/TAP
ff = open('/Users/silver/Box Sync/phd/pro/sne/sbo/wp4/coo.txt', 'w')
fn = open('/Users/silver/Box Sync/phd/pro/sne/sbo/wp4/ned.sh', 'w')
fn.write('#!/bin/bash\n\nrm ned.txt\nprintf \"# NED queries for all objects\n\" > ned.txt\n\n')
hlp = "curl \"https://ned.ipac.caltech.edu/tap/sync?query=SELECT+*+FROM+objdir+WHERE+CONTAINS(POINT('J2000',ra,dec),CIRCLE('J2000',{0:.12f},{1:.12f},0.002777777777777778))=1&LANG=ADQL&REQUEST=doQuery&FORMAT=text\" >> ned.txt\n"
for ii, ev in enumerate(coo):
    if ev[1] > 0:
        buf = '{0:16.12f} +{1:<15.12f}\n'.format(ev[0], ev[1])
    else:
        buf = '{0:16.12f} {1:16.12f}\n'.format(ev[0], ev[1])
    ff.write(buf)

    fn.write('printf \"\\n\\n{0:9d} {6:>30} {1:9.5f} {2:9.5f} {4:9.5f} {5:9.5f} {3:5.2f}\\n\" >> ned.txt\n'.format(obs[ii], coo[ii,0], coo[ii,1], coo[ii,2]*3600, gal[ii,0], gal[ii,1], co2[ii].to_string('hmsdms')))
    fn.write(hlp.format(ev[0], ev[1]))
    
ff.close()
fn.close()




################################################################
# Match Gaia data to EXTraS
off = gai['target_distance']
par = np.c_[gai['parallax'], gai['parallax_error']]
gid = gai['target_id']
counter = 0
seen = []

ff = open('/Users/silver/Box Sync/phd/pro/sne/sbo/wp4/gaia.txt', 'w')
ff.write('# GAIA objects matched to the EXTraS objects')
for ii, tar in enumerate(gid):
    if not tar in seen:
        counter += 1
        seen.append(tar)
        ff.write('\n{0:>9} {9:>30} {1:>9} {2:>9} {7:>9} {8:>9} {3:>5} {4:>6} {5:>8} {6:>6}\n'.format('OBS ID', 'RA', 'Dec', 'Error', 'Offset', 'Parallax', 'Error', 'l', 'b', 'hmsdms'))

    ra, dec = map(float,tar.split())
    c1 = SkyCoord(ra*u.deg, dec*u.deg, frame='icrs')
    jj = c1.separation(co2).argmin()
    ff.write('{0:9d} {9:>30} {1:9.5f} {2:9.5f} {7:9.5f} {8:9.5f} {3:5.2f} {4:6.2f} {5:8.2f} {6:6.2f}\n'.format(obs[jj], coo[jj,0], coo[jj,1], coo[jj,2]*3600, off[ii]*3600, par[ii,0], par[ii,1], gal[jj,0], gal[jj,1], co2[jj].to_string('hmsdms')))
ff.close()




################################################################
# Select some objects of interest by hand, preliminary
sel = (np.abs(gal[:,1]) > 15) # 39 objects out of the galactic plane
sel = (cts[:,0] > 75) & sel # 12 objects with more than 75 counts

for ii, it in enumerate(sel):
    if not it:
        continue

    print(dat[ii][:6],dat[ii][8:10])

# ('EXMM J004449.9+415244', '0109270301', 0.0, 1, 1, 3531.376999989152) (114.795425, 14.986726)
# One of the two sources in Andromeda

# ('EXMM J023126.0-712906', '0510181701', 0.353, 2, 1, 2176.972000002861) (144.08746, 18.107935)
# Seems to be associated with a star (parallax in Gaia)

# ('EXMM J031659.2-663214', '0405090101', 1.0, 3, 1, 1000.0) (116.7442, 11.711461)
# Seems to be associated with a star (parallax in Gaia). The offset is a bit large but the XMM image is close to a chip gap. Very crowded, close to bright star

# ('EXMM J053219.8-072932', '0690200201', 0.0, 2, 1, 2813.7109999656677) (142.71948, 16.605238)
# 3 Arcseconds away from an infrared source with parallax. Also close to chip gap

# ('EXMM J053521.8-055403', '0112660101', 1.0, 3, 1, 2000.0) (272.47998, 20.047096)
# Very close to Iota Orionis, mag 3 O star. Trigger in XMM looks instrumental

# ('EXMM J053546.1-051051', '0134531701', 0.986, 2, 1, 2000.0) (215.41632, 19.839794)
# Known flare star in Orion, V* V808 Ori, trigger in XMM looks instrumental

# ('EXMM J103528.6+631021', '0403760401', 0.414, 7, 1, 4660.324000000954) (157.72327, 22.439241)
# Close to SDSS galaxy
# http://skyserver.sdss.org/DR15//en/tools/explore/summary.aspx?id=1237654400224854184
# photoZ (KD-tree method) 0.177+-0.0222, util_distances.sh 0.177=855.593219 Mpc

# ('EXMM J104620.4+524822', '0200480201', 0.363, 4, 1, 921.5119999945164) (80.22472, 11.483685)
# Seems to be associated with a star (parallax in Gaia)

# ('EXMM J142517.6+225545', '0143652301', 0.0, 2, 1, 2012.2119999825954) (148.71573, 17.65371)
# Seems to be associated with a star (parallax in Gaia)

# ('EXMM J162714.7-245135', '0305540701', 0.0, 1, 1, 1000.0) (253.7473, 18.4499)
# Sits between two T tau type stars

# ('EXMM J162729.5-243917', '0305540701', 1.0, 3, 1, 5000.0) (151.27887, 14.937144)
# Very close to a young stellar object (YSO), a number from this region in EXTraS

# ('EXMM J212805.1-651052', '0670380101', 0.0, 3, 1, 1142.5189999938011) (99.716774, 12.72698)
# LEHPM 4113, High proper-motion Star
    
db()
