'''
2019-09-04, Dennis Alp, dalp@kth.se

Plot the ligth curves of the variable objects in EXTraS WP2 catalog of aperiodic variables. Most of the code is to retrieve and format catalog data for quick analysis of the light curves.
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
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.wcs import WCS
from astropy.io import fits
from astropy import units
from astroquery.vizier import Vizier
from astroquery.simbad import Simbad
from astroquery.ned import Ned

# For the catalogs
gaia_col = ['_r', 'RA_ICRS', 'DE_ICRS', 'Plx', 'e_Plx', 'Gmag', 'e_Gmag']
mass_col = ['_r', 'RAJ2000', 'DEJ2000', 'Hmag', 'e_Hmag'] 
sdss_col = ['_r', 'RA_ICRS', 'DE_ICRS', 'class', 'rmag', 'e_rmag', 'zsp', 'zph', 'e_zph', '__zph_']
Simbad._VOTABLE_FIELDS = ['distance_result', 'ra', 'dec', 'plx', 'plx_error','rv_value', 'z_value', 'otype(V)']
sdss_type = ['', '', '', 'galaxy', '', '', 'star']
sup_cos = ['', 'galaxy', 'star', 'unclassifiable', 'noise']



################################################################
#
def get_viz(coo):
    def have_gaia(tab):
        return (tab['Plx']/tab['e_Plx']).max() > 3

    viz = Vizier(columns=['*', '+_r'], timeout=60, row_limit=10).query_region(coo, radius=5*u.arcsec, catalog='I/345/gaia2,II/246/out,V/147/sdss12')
    nearest = 999.
    if len(viz) == 0:
        return '', False, nearest

    # Some formating
    tmp = []
    plx_star = False
    if 'I/345/gaia2' in viz.keys():
        tmp = tmp + ['Gaia DR2'] + viz['I/345/gaia2'][gaia_col].pformat(max_width=-1)
        plx_star = have_gaia(viz['I/345/gaia2'][gaia_col])
        nearest = np.min((viz['I/345/gaia2']['_r'].min(), nearest))
    if 'II/246/out' in viz.keys():
        tmp = tmp + ['\n2MASS'] + viz['II/246/out'][mass_col].pformat(max_width=-1)
        nearest = np.min((viz['II/246/out']['_r'].min(), nearest))
    if 'V/147/sdss12' in viz.keys():
        tmp = tmp + ['\nSDSS12'] + viz['V/147/sdss12'][sdss_col].pformat(max_width=-1)
        nearest = np.min((viz['V/147/sdss12']['_r'].min(), nearest))
    return '\n'.join(tmp), plx_star, nearest

def get_simbad(coo):
    def sim_hlp(coo, ts):
        # Recursive pauser to avoid blacklisting
        try:
            sim = Simbad.query_region(coo, radius=60*u.arcsec)
        except:
            print('\nSIMBAD blacklisted, pausing for', ts, 's.')
            time.sleep(ts)
            sim = sim_hlp(coo, 2*ts)
        return sim

    sim = sim_hlp(coo, 8)
    if sim is None or len(sim) == 0:
        return '', 999, False
    
    otype = sim[0]['OTYPE_V'].decode("utf-8").lower()
    if sim['DISTANCE_RESULT'][0] < 5.:
        if 'star' in otype or 'cv' in otype or 'stellar' in otype or 'spectroscopic' in otype or 'eclipsing' in otype or 'variable of' in otype:
            otype = True
        else:
            otype = False
    else:
        otype = False
    return '\n\nSIMBAD\n' + '\n'.join(sim.pformat(max_lines=-1, max_width=-1)[:13]), sim['DISTANCE_RESULT'].min(), otype
    
def get_ned(coo):
    def ned_hlp(coo, ts):
        # Recursive pauser to avoid blacklisting
        try:
            ned = Ned.query_region(coo, radius=60*u.arcsec)
        except:
            print('\nNED blacklisted, pausing for', ts, 's.')
            time.sleep(ts)
            ned = ned_hlp(coo, 2*ts)
        return ned

    ned = ned_hlp(coo, 8)
    if ned is None or len(ned) == 0:
        return '', 999

    # Remove 2MASS objects beyond 5 arcsec
    ned = ned['Separation', 'Object Name', 'RA', 'DEC', 'Type', 'Velocity', 'Redshift']
    ned['Separation'].convert_unit_to(u.arcsec)
    rm = []
    for jj, row in enumerate(ned):
        if row['Separation'] > 5. and row['Object Name'][:5] == '2MASS':
            rm.append(jj)
    ned.remove_rows(rm)
    if len(ned) == 0:
        return '', 999

    ned['Separation'] = ned['Separation'].round(3)
    ned['Separation'].unit = u.arcsec
    ned.sort('Separation')
    return '\n\n\nNED\n' + '\n'.join(ned.pformat(max_lines=-1, max_width=-1)[:13]), ned['Separation'].min()
    
def get_cla(coo):
    tmp = clc.separation(coo).arcsec
    # This compares coordinates from 3XMM-DR6 with EXTraS so there could be 'large' differences of around 1 arcsec.
    if tmp.min() < 3.: 
        return 'Class: {5:>5s}, Confidence: {6:>5.3f}, Method: {7:>13s}'.format(*cla[tmp.argmin()])
    else:
        return 'Unclassified by EXTraS WP7'
    


################################################################
# Parameters
figp = '/Users/silver/Box Sync/phd/pro/sne/sbo/wp2/fig/'
cla = fits.open('/Users/silver/Box Sync/phd/pro/sne/sbo/wp7/3xmm_classes.fits')[1].data
clc = SkyCoord(cla['src_ra']*u.deg, cla['src_dec']*u.deg, frame='icrs')
dat = fits.open('/Users/silver/Box Sync/phd/pro/sne/sbo/wp2/extras_aperiodic_light.fits')[1].data
qf = dat['QUALITY_FLAG'] # should be 0, else caution
dat = fits.open('/Users/silver/Box Sync/phd/pro/sne/sbo/wp2/extras_aperiodic.fits')[1].data
xf = dat['3XMM_SUM_FLAG'] # Clean < 3, else manually flagged
obsf = np.array(dat['OBS_ID'])
expf = np.array(dat['EXP_ID'])
camf = np.array(dat['CAMERA'])
numf = np.array(dat['SRC_NUM'])

# Prepare
files = sorted(glob('/Users/silver/Box Sync/phd/pro/sne/sbo/wp2/dat/*.fits'))
dat = []
fn = []
leg = []
nearest = np.zeros(3)
t0 = 0
start = 0
print(len(files), 'files')
for ii, ff in tqdm(enumerate(files[start:])):
    # Collect different light curve binnings for same detection
    dat.append(fits.open(ff))
    fn.append(ff.split('/')[-1])
    if ii+start+1 < len(files):
        obs = fn[-1].split('_')[0] == files[ii+start+1].split('/')[-1].split('_')[0]
        num = fn[-1].split('_')[1] == files[ii+start+1].split('/')[-1].split('_')[1]
        exp = fn[-1].split('_')[2] == files[ii+start+1].split('/')[-1].split('_')[2]
        cam = fn[-1].split('_')[3] == files[ii+start+1].split('/')[-1].split('_')[3]
        if obs and num and exp and cam:
            continue

    # Start plotting
    fig = plt.figure(figsize=(10, 7))
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    for jj, dd in enumerate(dat):
        # Read light curves
        if 'bb3' in fn[jj]:
            met = 'bb3'
            zorder = 10
            tt = np.zeros(2*dd[1].data['TSTART'].size)
            tt[::2]  = dd[1].data['TSTART']
            tt[1::2] = dd[1].data['TSTOP']
            ee = dd[1].data['ERATE']
        else:
            met = str(dd[1].header['TIMEDEL']).split('.')[0]
            tt = np.zeros(2*dd[1].data['TIME'].size)
            tt[::2]  = dd[1].data['TIME']-int(met)/2.+dd[1].header['TSTART']
            tt[1::2] = dd[1].data['TIME']+int(met)/2.+dd[1].header['TSTART']
            ee = dd[1].data['ERROR']
            if met == '500': zorder = 0
            elif float(met) > 500: zorder = 5
            else: zorder = -10

        obs = fn[jj].split('_')[0]
        cam = fn[jj].split('_')[1]
        exp = fn[jj].split('_')[2]
        num = fn[jj].split('_')[3]
        rr = zoom(dd[1].data['RATE'], 2, order=0)
        fe = dd[1].data['FRACEXP'] 

        if t0 == 0:
            t0 = tt[0]
        tt = tt-t0

        # Plot light curves
        if met == 'bb3' or float(met) > 50:
            tmp = ax1.plot(tt, rr, label=met, zorder=zorder)
            ax1_tmp = ax1.set_ylim(bottom=0)
            # Manual rescaling of ylim to prevent resizing to match error bars
            ax1_tmp = ax1.set_ylim(top=np.max((ax1_tmp[1], rr.max()*1.05)))
            ax1.errorbar((tt[::2]+tt[1::2])/2., rr[::2], yerr=ee, color=tmp[0].get_color(), fmt='.', ms=0, zorder=zorder)
            ax1.set_ylim(ax1_tmp)
            ax2.plot((tt[::2]+tt[1::2])/2., fe, 'x', color=tmp[0].get_color(), zorder=zorder)
        else:
            # Smooth very high cadence light curves
            zf = int(np.ceil(50/float(met)))
            rr = rr.reshape((rr.size//2, 2))[:,0]
            rr = rr[:rr.size//zf*zf]
            ee = ee[:ee.size//zf*zf]
            rr = rr.reshape((rr.size//zf, zf))
            ee = ee.reshape((ee.size//zf, zf))
            rr = np.average(rr, axis=1, weights=1/ee).repeat(2)
            tmp = ax1.plot(np.append(tt,tt[-1])[::2*zf].repeat(2)[1:-1], rr, label=str(int(zf*float(met))), zorder=zorder)
            ax2.plot((tt[::2]+tt[1::2])/2., fe, 'x', color=tmp[0].get_color(), zorder=zorder)

    # Cosmetics
    out = obs + '_' + cam + '_' + exp + '_' + num 
    ax1.set_title(out)
    ax1.legend()
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Rate (cts/s)')
    ax2.set_ylim([0, 1])
    ax2.set_ylabel('FRACEXP')

    # Get some meta data
    coo = SkyCoord(dd[1].header['SRC_RA']*u.deg, dd[1].header['SRC_DEC']*u.deg, frame='icrs')
    out_plane = np.abs(float(coo.galactic.to_string('decimal').split(' ')[1])) > 15
    viz, plx_star, nearest[0] = get_viz(coo)
    sim, nearest[1], otype = get_simbad(coo)
    ned, nearest[2] = get_ned(coo)
    tmp = ((obs==obsf) & (int(num) == numf) & (exp == expf) & (cam == camf)).argmax()
    xmmf, extf = xf[tmp], qf[tmp]

    # Print text to the figure
    plt.text(0, 0, viz + sim + ned, fontdict={'family': 'monospace', 'size': 12}, transform=plt.gcf().transFigure, va='top')
    buf = 'RA, Dec:' + '{0:>16s}{1:>16s}'.format(*coo.to_string('hmsdms').split(' '))+ '\n'
    buf = buf + 'RA, Dec:' + '{0:>15s},{1:>15s}'.format(*coo.to_string('decimal').split(' ')) + '\n'
    buf = buf + 'l, b:   ' + '{0:>15s},{1:>15s}'.format(*coo.galactic.to_string('decimal').split(' ')) + '\n'
    buf = buf + '3XMM Flag: {0:1d}, EXTraS Flag: {1:1d}'.format(xmmf, extf) + '\n'
    buf = buf + get_cla(coo)
    plt.text(0, 1, buf, fontdict={'family': 'monospace', 'size': 14}, transform=plt.gcf().transFigure, va='bottom')

    # Label, distribute into folders
    if xmmf > 2 and not extf:
        plt.savefig(figp + 'xmm_ext_flag/' + out + '.pdf', bbox_inches='tight', pad_inches=0.1, dpi=300)
    elif xmmf > 2:
        plt.savefig(figp + 'xmm_flag/' + out + '.pdf', bbox_inches='tight', pad_inches=0.1, dpi=300)
    elif not extf:
        plt.savefig(figp + 'ext_flag/' + out + '.pdf', bbox_inches='tight', pad_inches=0.1, dpi=300)
    elif nearest.min() > 5.:
        plt.savefig(figp + 'nothing_in_5_arcsec/' + out + '.pdf', bbox_inches='tight', pad_inches=0.1, dpi=300)
    elif out_plane and not plx_star:
        plt.savefig(figp + 'high_lat_no_plx_star/' + out + '.pdf', bbox_inches='tight', pad_inches=0.1, dpi=300)
    elif not plx_star:
        plt.savefig(figp + 'no_plx_star/' + out + '.pdf', bbox_inches='tight', pad_inches=0.1, dpi=300)
    elif otype:
        plt.savefig(figp + 'rest/star/' + out + '.pdf', bbox_inches='tight', pad_inches=0.1, dpi=300)
    else:
        plt.savefig(figp + 'rest/' + out + '.pdf', bbox_inches='tight', pad_inches=0.1, dpi=300)

    # Reset
    plt.close()
    dat = []
    fn = []
    leg = []
    nearest = np.zeros(3)
    t0 = 0
db()
