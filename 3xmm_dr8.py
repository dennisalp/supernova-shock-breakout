'''
2019-09-17, Dennis Alp, dalp@kth.se

Search for SBOs in the 3XMM DR8 catalog.
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
from astropy.time import Time

FNULL = open(os.devnull, 'w')
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
    return '\n\nNED\n' + '\n'.join(ned.pformat(max_lines=-1, max_width=-1)[:13]), ned['Separation'].min()
    


#For most uses of the catalog it is recommended to use either a detection flag (SUM_FLAG, EP_FLAG or SC_SUM_FLAG) or an observation flag (OBS_CLASS) as a filter to obtain what can be considered a 'clean' sample.

# There are 633733 out of 775153 detections that are considered to be clean (i.e., summary flag < 3).

#remove extended

#For 173208 detections, EPIC spectra and time series were automatically extracted during processing, and a χ2-variability test was applied to the time series. 5934 detections in the catalogue are considered variable, within the timespan of the specific observation, at a probability of 10-5 or less based on the null-hypothesis that the source is constant. Of these, 3603 have a summary flag  < 3.

# Exclude sources detected more than once
#DETID (K)
#A unique number which identifies each entry (detection) in the catalogue. The DETID numbering assignments in 3XMM-DR8 bear no relation to those in 3XMM-DR4 and earlier but the DETID of the matching detection from the 3XMM-DR4 catalogue to the 3XMM-DR8 detection is provided via the DR4DETID column.
#SRCID (K)
#A unique number assigned to a group of catalogue entries which are assumed to be the same source. The process of grouping detections in to unique sources has changed since the 2XMM catalogue series and is described in Section 3.8. The SRCID assignments in 3XMM-DR8 bear no relation to those in 3XMM-DR4 and earlier but the nearest unique sources from the 3XMM-DR4 catalogue to the 3XMM-DR8 unique source is provided via the DR4SRCID column

# To download data from server
#IAUNAME (21A)
#The IAU name assigned to the unique SRCID (see 3XMM-DR4 UG, Sec. 3.8.1 ).
#SRC_NUM (J) SAS task srcmatch
#The (decimal) source number in the individual source list for this observation; when expressed in the hexadecimal system it identifies the source-specific product files belonging to this detection (2XMM UG Appendix A.1).

#OBS_ID (10A)
#The XMM-Newton observation identification.
#REVOLUT (4A) [orbit]
#The XMM-Newton revolution number.
#MJD_START (D) [d]
#Modified Julian Date (i.e., JD - 2400000.5) of the start of the observation.
#OBS_CLASS (J)
#Quality classification of the whole observation based on the area flagged as bad in the manual flagging process as compared to the whole detection area, see 2XMM UG Sec. 3.2.6. 0 means nothing has been flagged; 1 indicates that 0% < area < 0.1% of the total detection mask has been flagged; 2 indicates that 0.1% <= area < 1% has been flagged; 3 indicates that 1% <= area < 10% has been flagged; 4 indicates that 10% <= area < 100% has been flagged; and 5 means that the whole field was flagged as bad.

#RA (D) [deg] SAS task catcorr
#Corrected Right Ascension of the detection in degrees (J2000) after statistical correlation of the emldetect coordinates, RA_UNC and DEC_UNC, with the USNO B1.0, 2MASS or SDSS (DR8) optical/IR source catalogues using the SAS task catcorr (the process of correcting the coordinates is also referred to as field rectification). In cases where the cross-correlation is determined to be unreliable, no correction is applied and this value is therefore the same as RA_UNC (cf. Sec. 3XMM-DR4 UG, Sec. 3.4).
#DEC (D) [deg]SAS task catcorr
#Corrected Declination of the detection in degrees (J2000) after statistical correlation of the emldetect coordinates, RA_UNC and DEC_UNC, with the USNO B1.0, 2MASS or SDSS (DR8) optical/IR source catalogues using the SAS task catcorr (the process of correcting the coordinates is also referred to as field rectification). In cases where the cross-correlation is determined to be unreliable no correction is applied and this value is therefore the same as DEC_UNC (cf. Sec. 3XMM-DR4 UG, Sec. 3.4).
#POSERR (E) [arcsec]
#Total radial position uncertainty in arcseconds calculated by combining the statistical error RADEC_ERR and the error arising from the field rectification process SYSERRCC as follows: POSERR = SQRT ( RADEC_ERR^2 + SYSERRCC^2 ). For a 2-dimensional Gaussian error distribution, this radius reflects a 63% probability that the true source position lies within this radius of the measured position. The corresponding 68% confidence radius is 1.0714*RADEC_ERR.
#LII (D) [deg]SAS task catcorr
#Galactic longitude of the detection in degrees corresponding to the (corrected) coordinates RA and DEC.
#BII (D) [deg]SAS task catcorr
#Galactic latitude of the detection in degrees corresponding to the (corrected) coordinates RA and DEC.

#REFCAT (I) SAS task catcorr
#An integer code reflecting the absolute astrometric reference catalogue which gave the statistically 'best' result for the field rectification process (from which the corrections are taken). It is 1 for the SDSS (DR9) catalogue, 2 for 2MASS and 3 for USNO B1.0. Where catcorr fails to produce a reliable solution, REFCAT is a negative number, indicating the cause of the failure. The failure codes are -1 = Too few matches (< 10), -2 = poor fit (goodness of fit parameter in catcorr < 5.0), -3 = error on the field positional rectification correction is > 0.75 arcseconds) (see Sec. 3XMM-DR4 UG, Sec. 3.4). .

#SUM_FLAG (J)
#The summary flag of the detection is derived from the EPIC flag (EP_FLAG, see 2XMM UG Sec. 3.1.2 h) and Sec. 3.2.6  but note also sections 3XMM-DR4 UG 3.11 and 3.7). It is 0 if none of the nine flags was set; it is set to 1 if at least one of the warning flags (flag 1, 2, 3, 9) was set but no possible-spurious-detection flag (flag 7, 8); it is set to 2 if at least one of the possible-spurious-detection flags (flag 7, 8) was set but not the manual flag (flag 11); it is set to 3 if the manual flag (flag 11) was set but no possible-spurious-detection flags (flag 7, 8); it is set to 4 if the manual flag (flag 11) as well as one of the possible-spurious-detection flags (flag 7, 8) is set. The meaning is thus: 
#0 = good, 
#1 = source parameters may be affected, 
#2 = possibly spurious, 
#3 = located in a area where spurious detection may occur, 
#4 = located in a area where spurious detection may occur and possibly spurious. 
#For details see 2XMM UG Sec. 3.2.7 but note that flag 12 is no longer used in 3XMM-DR4 (see section 3.11)

#TSERIES (L)
#The flag is set to True if this detection has a time series made in at least one exposure (cf. Sec. 3.6).

#SPECTRA (L)
#The flag is set to True if this detection has a spectrum made in at least one exposure (cf. Sec. 3.6).

#HIGH_BACKGROUND (L)
#The flag is set to True if this detection comes from a field which, during manual screening, was considered to have a high background level which notably impacted on source detection (see Sec. 6.1.2). This column is new in 3XMM-DR4.

#EP_CHI2PROB (E)
#The χ2 probability (based on the null hypothesis) that the time series of the source, as detected by any of the cameras, can be explained by a constant flux. The minimum value of the available camera probabilities (PN_CHI2PROB, M1_CHI2PROB, M2_CHI2PROB) is given.

#VAR_FLAG (L)
#The flag is set to True if this source was detected as variable (χ2 probability < 1E-5, see PN_CHI2PROB, M1_CHI2PROB, M2_CHI2PROB) in at least one exposure (see 2XMM UG Sec. 3.2.8). Note that where a timeseries is not available or insufficient points are left in the timeseries after applying background flare GTIs, the value is set to NULL or Undefined

#VAR_EXP_ID (4A)
#The exposure ID ('S' or 'U' followed by a three-digit number) of the exposure with the smallest χ2 probability is given here.

#VAR_INST_ID (2A)
#The instrument ID (PN, M1, M2) of the exposure given in VAR_EXP_ID is listed here.

#MJD_FIRST (D) [d]
#The MJD start date (MJD_START) of the earliest observation of any constituent detection of the unique source.
#MJD_LAST (D) [d]
#The MJD end date (MJD_STOP) of the last observation of any constituent detection of the unique source.
#N_DETECTIONS (J)
#The number of detections of the unique source SRCID used to derive the combined values.
#CONFUSED (J)
#Normally set False, but set True when a given detection has a probability above zero of being associated with two or more distinct sources. The SRCID is that of the match with the highest probability, but there remains some uncertainty about which source is the correct match for the detection.



################################################################
# Just read the data
figp = '/Users/silver/Box Sync/phd/pro/sne/sbo/3xmm_dr8/fig/'
dat = fits.open('/Users/silver/Box Sync/phd/pro/sne/sbo/3xmm_dr8/3XMM_DR8cat_v1.0.fits')[1].data
cwd = '/Users/silver/Box Sync/phd/pro/sne/sbo/3xmm_dr8/dat/'
os.chdir(cwd)
obs_num = '{0:10s}_{1:03X}'
url = 'http://nxsa.esac.esa.int/nxsa-sl/servlet/data-action-aio?level=PPS&extension=FTZ&name=SRCTSR&obsno={0:10s}&sourceno={1:03X}'
#print(dat.dtype.names)

detid = dat['DETID']
srcid = dat['SRCID']
iauid = dat['IAUNAME']
num = dat['SRC_NUM']
obs = dat['OBS_ID']
rev = dat['REVOLUT']
t0 = dat['MJD_START']
t1 = dat['MJD_STOP']
ra = dat['RA']
de = dat['DEC']
err = dat['POSERR']
ll = dat['LII']
bb = dat['BII']
ref = dat['REFCAT']
pval = dat['EP_CHI2PROB']
flag_var = dat['VAR_FLAG']
flag_obs = dat['OBS_CLASS']
flag_sum = dat['SUM_FLAG']
flag_ts = dat['TSERIES']
flag_spec = dat['SPECTRA']
flag_bkg = dat['HIGH_BACKGROUND']
var_exp = dat['VAR_EXP_ID']
var_cam = dat['VAR_INST_ID']
first = dat['MJD_FIRST']
last = dat['MJD_LAST']
ndet = dat['N_DETECTIONS']
conf = dat['CONFUSED']
ext = dat['SC_EXTENT']
cts = dat['EP_8_CTS']
cts_err = dat['EP_8_CTS_ERR']
rate = dat['EP_8_RATE']
rate_err = dat['EP_8_RATE_ERR']
flux = dat['EP_8_FLUX']
flux_err = dat['EP_8_FLUX_ERR']



################################################################
#
gg = flag_var & (flag_sum < 3)
tot = gg.sum()
counter = 0
nearest = np.zeros(3)
tref = 0
retcode = subprocess.call(['mkdir', '-p', figp + '1_extended'])
retcode = subprocess.call(['mkdir', '-p', figp + '2_star'])
retcode = subprocess.call(['mkdir', '-p', figp + '3_persistent'])
retcode = subprocess.call(['mkdir', '-p', figp + '4_nothing_in_5_arcsec'])
retcode = subprocess.call(['mkdir', '-p', figp + '5_high_lat_no_plx_star'])
retcode = subprocess.call(['mkdir', '-p', figp + '6_no_plx_star'])
retcode = subprocess.call(['mkdir', '-p', figp + '7_rest'])

for ii, gd in enumerate(gg[:]):
    if not gd:
        continue
    counter += 1
    if np.mod(counter, 30) == 1:
        print('{0:4d}, {1:5.3f}'.format(counter, counter/tot))
    out = obs_num.format(obs[ii], num[ii])
    
    # Query catalogs
    coo = SkyCoord(ra[ii]*u.deg, de[ii]*u.deg, frame='icrs')
    out_plane = np.abs(bb[ii]) > 15
    viz, plx_star, nearest[0] = get_viz(coo)
    sim, nearest[1], otype = get_simbad(coo)
    ned, nearest[2] = get_ned(coo)

    # Parse meta data
    buf = 100*'#' + '\nCatalog entry ' + str(ii) + ', Number ' + str(counter) + '\n' + 100*'#' + '\n\n'
    buf = buf + 'IAU ID: ' + iauid[ii] + ', Obs. Date: ' + Time(t0[ii], format='mjd').iso + '\n'
    buf = buf + 'Obs. ID: ' + obs[ii] + ', Source number: ' + str(num[ii]) + '({0:03X}'.format(num[ii]) + ')\n'
    buf = buf + 'Detection ID: ' + str(detid[ii]) + ', Source ID: ' + str(srcid[ii]) + '\n'
    buf = buf + 'Detections: {0:2d}'.format(ndet[ii]) + ', First: ' + Time(first[ii], format='mjd').iso + '\n'
    buf = buf + 16*' ' + 'Last:  ' + Time(last[ii], format='mjd').iso + '\n\n'
    buf = buf + 'RA, Dec:' + '{0:>16s}{1:>16s}'.format(*coo.to_string('hmsdms').split(' '))+ '\n'
    buf = buf + 'RA, Dec:' + '{0:>16s}{1:>16s}'.format(*coo.to_string('decimal').split(' ')) + '\n'
    buf = buf + 'l, b:   ' + '{0:>16s}{1:>16s}'.format(*coo.galactic.to_string('decimal').split(' ')) + '\n\n'
    buf = buf + 'Counts: {0:7.0f}({1:.0f})'.format(cts[ii], cts_err[ii]) + ', Exposure: {0:3.0f} ks\n'.format((t1[ii]-t0[ii])*24*3.6)
    buf = buf + 'Flux: ' + str(flux[ii]) + '(' + str(flux_err[ii]) + '), Count rate: ' + str(rate[ii]) + '(' + str(rate_err[ii]) + ')\n\n'

    

    ################################################################
    # Download and plot light curves
    fig = plt.figure(figsize=(10, 7))
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    # Dowload data
    retcode = subprocess.call(['rm', '-rf', cwd + out])
    retcode = subprocess.call(['rm'] + glob('GUEST*.tar'), stderr=subprocess.DEVNULL)
    retcode = subprocess.call(['rm'] + glob('*.FTZ'), stderr=subprocess.DEVNULL)
    cmd = ['curl', '-sS', '-O', '-J']
    retcode = subprocess.call(cmd + [url.format(obs[ii], num[ii])])
    gtar = glob('GUEST*.tar')
    gftz = glob('*.FTZ')

    if len(gtar) == 1:
        retcode = subprocess.call(['tar', '-xvf', cwd + gtar[0], '-C', cwd], stderr=subprocess.DEVNULL)
        retcode = subprocess.call(['mv', cwd + obs[ii] + '/pps', cwd + out])
        retcode = subprocess.call(['rm', '-rf', cwd + obs[ii]])
        retcode = subprocess.call(['rm', cwd + gtar[0]])
    elif len(gftz) == 1:
        retcode = subprocess.call(['mkdir', cwd + out])
        retcode = subprocess.call(['mv', cwd + gftz[0], cwd + out])
    else:
        print('ERROR Unknown data format delivered from XSA AIO')
        sys.exit(1)

    for ff in glob(cwd + out + '/*FTZ'):
        cam = ff.split('/')[-1][11:13]
        if cam == 'R1' or cam == 'R2':
            continue

        # Load data
        dd = fits.open(ff)[1]
        tbin = dd.header['TIMEDEL']
        tt = np.zeros(2*dd.data['TIME'].size)
        tt[::2]  = dd.data['TIME']-tbin/2.
        tt[1::2] = dd.data['TIME']+tbin/2.
        if tref < 1e-10:
            tref = tt[0]
        tt = tt-tref

        rr = zoom(dd.data['RATE'], 2, order=0)
        rr[np.isnan(rr)] = 0.
        ee = dd.data['ERROR']
        ee[np.isnan(ee)] = 1.e17
        ee[ee==0.] = 1.e17
        fe = dd.data['FRACEXP']

        # Plot data
        if tbin > 50:
            tmp = ax1.plot(tt, rr, label=cam + ' ' + str(tbin))
            ax1_tmp = ax1.set_ylim(bottom=0)
            # Manual rescaling of ylim to prevent resizing to match error bars
            ax1_tmp = ax1.set_ylim(top=np.max((ax1_tmp[1], rr.max()*1.05)))
            ax1.errorbar((tt[::2]+tt[1::2])/2., rr[::2], yerr=ee, color=tmp[0].get_color(), fmt='.', ms=0)
            ax1.set_ylim(ax1_tmp)
            ax2.plot((tt[::2]+tt[1::2])/2., fe, 'x', color=tmp[0].get_color())
        else:
            # Smooth very high cadence light curves
            zf = int(np.ceil(50/tbin))
            rr = rr.reshape((rr.size//2, 2))[:,0]
            rr = rr[:rr.size//zf*zf]
            ee = ee[:ee.size//zf*zf]
            rr = rr.reshape((rr.size//zf, zf))
            ee = ee.reshape((ee.size//zf, zf))
            rr = np.average(rr, axis=1, weights=1/ee).repeat(2)
            tmp = ax1.plot(np.append(tt,tt[-1])[::2*zf].repeat(2)[1:-1], rr, label=cam + ' ' + str(int(zf*tbin)))
            ax1.set_ylim(bottom=0)
            ax2.plot((tt[::2]+tt[1::2])/2., fe, 'x', color=tmp[0].get_color())


        
    # Cosmetics
    ax1.set_title(out)
    ax1.legend()
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Rate (cts/s)')
    ax2.set_ylim([0, 1])
    ax2.set_ylabel('FRACEXP')

    plt.text(0, 0, viz + sim + ned, fontdict={'family': 'monospace', 'size': 12}, transform=plt.gcf().transFigure, va='top')
    plt.text(0, 1, buf, fontdict={'family': 'monospace', 'size': 14}, transform=plt.gcf().transFigure, va='bottom')

    # Label, distribute into folders
    if ext[ii] > 1e-10:
        plt.savefig(figp + '1_extended/' + out + '.pdf', bbox_inches='tight', pad_inches=0.1, dpi=300)
    elif otype:
        plt.savefig(figp + '2_star/' + out + '.pdf', bbox_inches='tight', pad_inches=0.1, dpi=300)
    elif ndet[ii] > 1 and last[ii]-first[ii] > 3:
        plt.savefig(figp + '3_persistent/' + out + '.pdf', bbox_inches='tight', pad_inches=0.1, dpi=300)
    elif nearest.min() > 5.:
        plt.savefig(figp + '4_nothing_in_5_arcsec/' + out + '.pdf', bbox_inches='tight', pad_inches=0.1, dpi=300)
    elif out_plane and not plx_star:
        plt.savefig(figp + '5_high_lat_no_plx_star/' + out + '.pdf', bbox_inches='tight', pad_inches=0.1, dpi=300)
    elif not plx_star:
        plt.savefig(figp + '6_no_plx_star/' + out + '.pdf', bbox_inches='tight', pad_inches=0.1, dpi=300)
    else:
        plt.savefig(figp + '7_rest/' + out + '.pdf', bbox_inches='tight', pad_inches=0.1, dpi=300)

    plt.close()
    nearest = np.zeros(3)
    tref = 0
    
db()
