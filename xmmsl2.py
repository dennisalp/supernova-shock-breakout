'''
2019-09-16, Dennis Alp, dalp@kth.se

Search for SBOs in the XMM slew survey.
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



################################################################
# Just read the data
dat = fits.open('/Users/silver/Box Sync/phd/pro/sne/sbo/xmmsl2/xmmsl2_total.fits')[1].data
#print(dat.dtype.names)
name = dat['UNIQUE_SRCNAME']
obs = dat['OBSID']
src = dat['SOURCENUM']
ra = dat['RA']
de = dat['DEC']
re = dat['RADEC_ERR']
ll = dat['LII']
bb = dat['BII']
hr = dat['HR1']
hre = dat['HR1_ERR']
tt = dat['DATE_OBS']

cts8 = dat['SCTS_B8']
#The number of background subtracted counts, in the total energy band (0.2-12 keV).
#This number has been corrected for photons scattered outside the source region due to the Point Spread Function (PSF).
cts8e = dat['SCTS_B8_ERR']
#Statistical 1 sigma error on the total band source counts.
cts7 = dat['SCTS_B7']
#The number of background subtracted counts, in the hard energy band (2-12 keV), corrected for the PSF. Units: counts
cts7e = dat['SCTS_B7_ERR']
#Statistical 1 sigma error on the hard band source counts.
cts6 = dat['SCTS_B6']
#The number of background subtracted counts, in the soft energy band (0.2-2 keV), corrected for the PSF. Units: counts
cts6e = dat['SCTS_B6_ERR']
#Statistical 1 sigma error on the soft band source counts.

ext8 = dat['EXT_B8']
#The spatial extension of the source in the total energy band.
#Units: 4.1"x4.1" image pixels
#This measures the deviation from a point source of the spatial distribution of the source counts. It is defined as the sigma of a Gaussian which would need to be convolved with the point spread function (PSF) to produce the observed counts distribution. The software (emldetect) fits sources out to a maximum extent of 20 pixels.
ext8e = dat['EXT_B8_ERR']
#Statistical one sigma error on the total band extension parameter.
ext7 = dat['EXT_B7']
#Spatial extension of the source in the hard energy band. Units: 4.1"x4.1" image pixels
ext7e = dat['EXT_B7_ERR']
#Statistical one sigma error on the hard band extension parameter.
ext6 = dat['EXT_B6']
#Spatial extension of the source in the soft energy band.
#Units: 4.1"x4.1" image pixels
ext6e = dat['EXT_B6_ERR']
#Statistical one sigma error on the soft band extension parameter.

flag = dat['VAL_FLAG']
#A text string which is set to 'CLEAN_SAMPLE' if this source is included in the clean subset. Otherwise the string is set to 'XXXXXXXXXXXX'
att = dat['VER_PSUSP']

id1 = dat['IDENT']
#Cross-correlations of the positions of the slew sources with astronomical databases and catalogues have been performed (see section on IDs). This column gives the catalogue name of the best match.
id2 = dat['ALTIDENT']
#An alternative name for the best match source.
id_rass = dat['RASSNAME']
#The name of the closest Rosat All Sky Survey (RASS) source.
otype = dat['ID_CATEGORY']
#The source type as returned by SIMBAD, NED and the other resources used in the cross-matching process. This is directly taken from the catalogue in question and no attempt has been made to rationalise the values.
cat = dat['ID_RESOURCE']
#The astronomical database or catalogue from which the best match has been selected. e.g. SIMBAD, NED, etc.
rd = dat['ID_DIST']
#The distance in arcminutes between the best match candidate and the slew survey source.
#Units: arcseconds
rr = dat['RASS_DIST']
#The distance from the best match ROSAT source and the slew source. Units: arcseconds




################################################################
# 
counter = 0
for ii in range(0, dat.size):
    if dat['VAL_FLAG'][ii] == 'CLEAN_SAMPLE' and att[ii] == False and np.isnan(rr[ii]) and np.abs(bb[ii]) > 15:
        ext = False
        if not np.isnan(ext6[ii]) and not ext6[ii] == 0. and ext6[ii]/ext6e[ii] > 3:
            ext = True
        if not np.isnan(ext7[ii]) and not ext7[ii] == 0. and ext7[ii]/ext7e[ii] > 3:
            ext = True
        if not np.isnan(ext8[ii]) and not ext8[ii] == 0. and ext8[ii]/ext8e[ii] > 3:
            ext = True
        if not ext:
            cts = 0.
            if not np.isnan(cts6[ii]) and cts6[ii]/cts6e[ii] > 3:
                cts += cts6[ii]
            if not np.isnan(cts7[ii]) and cts7[ii]/cts7e[ii] > 3:
                cts += cts7[ii]
            if not np.isnan(cts8[ii]) and cts8[ii]/cts8e[ii] > 3:
                cts += cts8[ii]
            if cts > 100.:
                counter += 1
                print('\n' + 100*'-' + '\n{0:<4d} {1:25s} {2:10s} {3:3d}'.format(counter, name[ii], obs[ii], src[ii]))
                if de[ii] >= 0.:
                    print('{0:9.5f} +{1:<8.5f} ({2:<6.3})'.format(ra[ii], de[ii], re[ii]))
                else:
                    print('{0:9.5f} {1:<9.5f} ({2:<6.3})'.format(ra[ii], de[ii], re[ii]))
                print('{0:9.5f} {1:<9.5f}'.format(ll[ii], bb[ii]))
                if np.isnan(rd[ii]):
                    print('2RXS ' + id_rass[ii] + '\n' + str(rr[ii]) + ' arcsec')
                else:
                    print(id1[ii] + '\n2RXS ' + id_rass[ii] + '\n' + otype[ii] + '\n' + cat[ii] + '\n' + str(rd[ii]) + ' arcseconds\n' + str(rr[ii]) + ' arcsec')

                print(tt[ii], cts, 'counts')
#db()
