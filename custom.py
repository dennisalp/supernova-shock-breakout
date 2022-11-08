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
import scipy.stats as sts
from scipy.ndimage import zoom
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.wcs import WCS
from astropy.io import fits
from astropy import units
from astroquery.vizier import Vizier
from astroquery.simbad import Simbad
from astroquery.ned import Ned
from astropy.time import Time

# For the catalogs
gaia_col = ['_r', 'RA_ICRS', 'DE_ICRS', 'Plx', 'e_Plx', 'Gmag', 'e_Gmag']
mass_col = ['_r', 'RAJ2000', 'DEJ2000', 'Hmag', 'e_Hmag'] 
sdss_col = ['_r', 'RA_ICRS', 'DE_ICRS', 'class', 'rmag', 'e_rmag', 'zsp', 'zph', 'e_zph', '__zph_']
Simbad._VOTABLE_FIELDS = ['distance_result', 'ra', 'dec', 'plx', 'plx_error','rv_value', 'z_value', 'otype(V)']
sdss_type = ['', '', '', 'galaxy', '', '', 'star']
sup_cos = ['', 'galaxy', 'star', 'unclassifiable', 'noise']
bad_types = ['star', 'cv', 'stellar', 'spectroscopic', 'eclipsing', 'variable of', 'binary', 'white', 'cepheid', 'dwarf']
agn_types = ['seyfert', 'quasar', 'blazar', 'bl lac', 'liner', 'active']



################################################################
#
class Observation:
    def __init__(self, ii):
        self.xx = np.empty(0)
        self.yy = np.empty(0)
        self.ii = ii
        for ff in glob(cwd + obs[self.ii] + '/*FTZ'):
            self.cam = ff.split('/')[-1][11:13]
            dd = fits.open(ff)[1].data

            # Concerning flags
            # https://xmm-tools.cosmos.esa.int/external/sas/current/doc/eimageget/node4.html
            # http://xmm.esac.esa.int/xmmhelp/EPICpn?id=8560;expression=xmmea;user=guest
            if self.cam == 'PN':
                good = dd['FLAG'] <= XMMEA_EP
                good = good & (dd['PATTERN'] <= 4)
            elif self.cam == 'M1' or self.cam == 'M2':
                good = dd['FLAG'] <= XMMEA_EM
                good = good & (dd['PATTERN'] <= 12)
            else:
                print('ERROR Unknown instrument:', cam, ii)
                sys.exit(1)
                
            good = good & (dd['PI'] > 300.) & (dd['PI'] < 10000.)
#            good = good & (dd['X'] > -99999998) # XMM puts some invalid photons at (-99999999.0, -99999999.0)
#            good = good & (dd['Y'] > -99999998)
            good = good & (dd['X'] > 0.) # XMM puts some invalid photons at (-99999999.0, -99999999.0)
            good = good & (dd['Y'] > 0.)
            self.xx = np.concatenate((self.xx, dd['X'][good]))
            self.yy = np.concatenate((self.yy, dd['Y'][good]))

        self.wcs = get_wcs(ff)
        self.img, self.xbin, self.ybin = np.histogram2d(self.xx, self.yy, img_size)


        
class Xoi:
    def __init__(self, ii):
        # Query catalogs
        self.nearest = np.empty(3)
        viz, self.plx_star, self.nearest[0] = get_viz(coo[ii])
        sim, self.nearest[1], self.good_type, self.agn = get_sim(coo[ii])
        ned, self.nearest[2] = get_ned(coo[ii])

        # Parse meta data
        buf = 100*'#' + '\nIndex ' + str(ii) + '\n' + 100*'#' + '\n\n'
        buf = buf + 'IAU ID: ' + iauid[ii] + ', Obs. Date: ' + Time(t0[ii], format='mjd').iso + '\n'
        buf = buf + 'Obs. ID: ' + obs[ii] + ', Source number: ' + str(num[ii]) + '({0:03X}'.format(num[ii]) + ')\n'
        buf = buf + 'Detection ID: ' + str(detid[ii]) + ', Source ID: ' + str(srcid[ii]) + '\n'
        buf = buf + 'Detections: {0:2d}'.format(ndet[ii]) + ', First: ' + Time(first[ii], format='mjd').iso + '\n'
        buf = buf + 16*' ' + 'Last:  ' + Time(last[ii], format='mjd').iso + '\n\n'
        buf = buf + 'RA, Dec:' + '{0:>16s}{1:>16s}'.format(*coo[ii].to_string('hmsdms').split(' '))+ '\n'
        buf = buf + 'RA, Dec:' + '{0:>16s}{1:>16s}'.format(*coo[ii].to_string('decimal').split(' ')) + '\n'
        buf = buf + 'l, b:   ' + '{0:>16s}{1:>16s}'.format(*coo[ii].galactic.to_string('decimal').split(' ')) + '\n\n'
        buf = buf + 'Counts: {0:7.0f}({1:.0f})'.format(cts[ii], cts_err[ii]) + ', Exposure: {0:3.0f} ks, '.format((t1[ii]-t0[ii])*24*3.6) + 'Flux: ' + str(flux[ii]) + '(' + str(flux_err[ii]) + '), Count rate: ' + str(rate[ii]) + '(' + str(rate_err[ii]) + ')\n'
        buf = buf + 'Null hypothesis probability: {0:7.1e}, Fractional excess above 90th percentile: {1:7.2f}\n\n'

        self.ii = ii
        self.out = obs_num.format(obs[ii], num[ii])
        self.viz = viz
        self.sim = sim
        self.ned = ned
        self.buf = buf
        self.persistent = last[ii]-first[ii] > 2.
        self.src_tt = np.empty(0)
        self.src_pi = np.empty(0)
        self.bkg_tt = np.empty(0)
        self.bkg_pi = np.empty(0)
        if ext[ii] >= 1.e-5:
            self.src_rad = np.max((400, ext[ii]/0.05)) # pixels, pixel size of 50 mas
            self.bkg_rad = [np.max((1000, self.src_rad)), np.max((2000, 2*self.src_rad))] # "To avoid non-converging fitting an upper limit of 80" is imposed."
        else:
            self.src_rad = 400 # pixels, pixel size of 50 mas
            self.bkg_rad = [1000, 2000]
        self.backscal = self.src_rad**2/(self.bkg_rad[1]**2-self.bkg_rad[0]**2)

    def keep(self):
        return self.good_type and not self.plx_star

    def ext_evt(self, wcs):
        for ff in glob(cwd + obs[self.ii] + '/*FTZ'):
            self.cam = ff.split('/')[-1][11:13]
            xx, yy = np.array(wcs.all_world2pix(coo[self.ii].ra, coo[self.ii].dec, 1))
            self.coo = np.array([xx, yy])
            dd = fits.open(ff)[1].data

            if self.cam == 'PN':
                good = dd['FLAG'] <= XMMEA_EP
                good = good & (dd['PATTERN'] <= 4)
            elif self.cam == 'M1' or self.cam == 'M2':
                good = dd['FLAG'] <= XMMEA_EM
                good = good & (dd['PATTERN'] <= 12)
            
            good = good & (dd['PI'] > 200.) & (dd['PI'] < 10000)
            rad = (dd['X']-xx)**2+(dd['Y']-yy)**2

            src = rad < self.src_rad**2
            src = good & src
            bkg = (rad > self.bkg_rad[0]**2) & (rad < self.bkg_rad[1]**2)
            bkg = good & bkg

            self.src_tt = np.concatenate((self.src_tt, dd['TIME'][src]))
            self.src_pi = np.concatenate((self.src_pi, dd['PI'][src]/1e3))
            self.bkg_tt = np.concatenate((self.bkg_tt, dd['TIME'][bkg]))
            self.bkg_pi = np.concatenate((self.bkg_pi, dd['PI'][bkg]/1e3))

        order = np.argsort(self.src_tt)
        self.src_tt = self.src_tt[order]
        self.src_pi = self.src_pi[order]
        order = np.argsort(self.bkg_tt)
        self.bkg_tt = self.bkg_tt[order]
        self.bkg_pi = self.bkg_pi[order]

    def mk_img(self, observation):
        xbin = np.arange(self.coo[0]-self.bkg_rad[1], self.coo[0]+self.bkg_rad[1]+1e-3, binning)
        ybin = np.arange(self.coo[1]-self.bkg_rad[1], self.coo[1]+self.bkg_rad[1]+1e-3, binning)
        self.img, _, _ = np.histogram2d(observation.xx, observation.yy, [xbin, ybin])
#        plt.imshow(gaussian_filter(np.log10(observation.img+1),2).T, vmin=0, vmax=np.percentile(gaussian_filter(np.log10(observation.img+1), 2), 99.9), cmap='afmhot', origin='lower')

    def mk_lc(self):
        self.tlc = self.src_tt[::cts_per_bin]
        self.src_lc = cts_per_bin/np.diff(self.tlc)
        self.bkg_lc, _ = np.histogram(self.bkg_tt, self.tlc)
        self.bkg_lc = self.backscal*self.bkg_lc/np.diff(self.tlc)
        self.sub_lc = self.src_lc - self.bkg_lc
        
        self.chi2 = np.sum((self.sub_lc-self.sub_lc.mean())**2/(self.src_lc/np.sqrt(cts_per_bin))**2) # Ignore uncertainty in background
        self.pval = sts.chi2.sf(self.chi2, self.sub_lc.size-1)
        self.fea90p = self.sub_lc.max()/np.percentile(self.sub_lc, 90)
        self.buf = self.buf.format(self.pval, self.fea90p)
        self.extremely_flaring = self.fea90p > extremely_flaring
        self.very_flaring = self.fea90p > very_flaring
        self.flaring = self.fea90p > flaring

    def mk_spec(self):
        self.src_spec, _ = np.histogram(self.src_pi, spec_bin)
        self.bkg_spec, _ = np.histogram(self.bkg_pi, spec_bin)
        self.src_spec = self.src_spec/np.diff(spec_bin)
        self.bkg_spec = self.backscal*self.bkg_spec/np.diff(spec_bin)

    def mk_postcard(self):
        fig = plt.figure(figsize=(25, 4))
        gs.GridSpec(1,5)
#        fig.subplots_adjust(hspace = 0.5)

        # Light curve
        ax = plt.subplot2grid((1,5), (0,0), colspan=2, rowspan=1)
        tmp = np.empty(2*self.tlc.size-2)
        tmp[:-1:2] = self.tlc[:-1]
        tmp[1::2] = self.tlc[1:]
        shift = int(tmp[0]//1000*1000)
        tmp = tmp - shift
        plt.plot(tmp, self.src_lc.repeat(2), label='Source', color='greenyellow')
        plt.plot(tmp, self.bkg_lc.repeat(2), label='Background', color='gold')
        plt.plot(tmp, self.sub_lc.repeat(2), label='Subtracted', color='k')
        ax.set_title(self.out)
        ax.legend()
        ax.set_xlabel('Time-{0:d} (s)'.format(shift))
        ax.set_ylabel('Rate (cts/s)')
        ax.set_ylim(bottom=0)

        # Spectrum
        ax = plt.subplot2grid((1,5), (0,2), colspan=1, rowspan=1)
        tmp = np.empty(2*spec_bin.size-2)
        tmp[:-1:2] = spec_bin[:-1]
        tmp[1::2] = spec_bin[1:]
        plt.loglog(tmp, self.src_spec.repeat(2), label='Source')
        plt.loglog(tmp, self.bkg_spec.repeat(2), label='Background')
        ax.legend()
        ax.set_xlabel('Energy (keV)'.format(shift))
        ax.set_ylabel('Flux density (cts/keV)')
        
        # Zoomed image
        ax = plt.subplot2grid((1,5), (0,3), colspan=1, rowspan=1)
        plt.imshow(np.log10(self.img.T+1), cmap='afmhot', origin='lower')
        tmp = plt.Circle((self.bkg_rad[1]/binning-0.5, self.bkg_rad[1]/binning-0.5), self.src_rad/binning, color='w', lw=2, fill=False)
        ax.add_artist(tmp)
        tmp = plt.Circle((self.bkg_rad[1]/binning-0.5, self.bkg_rad[1]/binning-0.5), self.bkg_rad[0]/binning, color='w', lw=1, fill=False)
        ax.add_artist(tmp)
        tmp = plt.Circle((self.bkg_rad[1]/binning-0.5, self.bkg_rad[1]/binning-0.5), self.bkg_rad[1]/binning, color='w', lw=1, fill=False)
        ax.add_artist(tmp)
        ax.set_axis_off()

        # Image
        ax = plt.subplot2grid((1,5), (0,4), colspan=1, rowspan=1)
        tmp = np.log10(observation.img+1).T
        tmp = gaussian_filter(tmp,2)
        plt.imshow(tmp, vmin=np.percentile(tmp, 60.), vmax=np.percentile(tmp, 99.9), cmap='afmhot', origin='lower')

        tmp = float(griddata(observation.xbin, np.arange(0,img_size+1), self.coo[0]))-0.5
        tmp = [tmp, float(griddata(observation.ybin, np.arange(0,img_size+1), self.coo[1]))-0.5]
        tmp = plt.Circle(tmp, 40, color='greenyellow', lw=2, fill=False)
        ax.add_artist(tmp)
        ax.set_axis_off()

        # Meta
        plt.text(0, 0, self.viz + self.sim + self.ned, fontdict={'family': 'monospace', 'size': 12}, transform=plt.gcf().transFigure, va='top')
        plt.text(0, 1, self.buf, fontdict={'family': 'monospace', 'size': 14}, transform=plt.gcf().transFigure, va='bottom')
        
        # Label, distribute into folders
        tmp = cwd
        tmp += self.out + '.pdf'
        plt.savefig(tmp, bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.show()
        np.savetxt(tmp.replace('.pdf', '.txt'), np.c_[self.src_lc, self.bkg_lc, self.sub_lc])
        return 1
   
def get_wcs(ff):
    dd = fits.open(ff)
    wcs = WCS()
    wcs.wcs.crpix = [dd[0].header['REFXCRPX'], dd[0].header['REFYCRPX']]
    wcs.wcs.cdelt = [dd[0].header['REFXCDLT'], dd[0].header['REFYCDLT']]
    wcs.wcs.crval = [dd[0].header['REFXCRVL'], dd[0].header['REFYCRVL']]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return wcs

def get_viz(coo):
    def have_gaia(tab):
        return (tab['Plx']/tab['e_Plx']).max() > 3

    def viz_hlp(coo, ts):
        # Recursive pauser to avoid blacklisting
        try:
            viz = Vizier(columns=['*', '+_r'], timeout=60, row_limit=10).query_region(coo, radius=5*u.arcsec, catalog='I/345/gaia2,II/246/out,V/147/sdss12')
        except:
            print('Vizier blacklisted, pausing for', ts, 's.')
            time.sleep(ts)
            viz = viz_hlp(coo, 2*ts)
        return viz

    viz = viz_hlp(coo, 8)
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

def get_sim(coo):
    def sim_hlp(coo, ts):
        # Recursive pauser to avoid blacklisting
        try:
            sim = Simbad.query_region(coo, radius=60*u.arcsec)
        except:
            print('SIMBAD blacklisted, pausing for', ts, 's.')
            time.sleep(ts)
            sim = sim_hlp(coo, 2*ts)
        return sim

    sim = sim_hlp(coo, 8)
    if sim is None or len(sim) == 0:
        return '', 999, True, False
    
    otype_v = sim[0]['OTYPE_V'].decode("utf-8").lower()
    if sim['DISTANCE_RESULT'][0] < 5.:
        good_type = not any(bt in otype_v for bt in bad_types)
        agn = any(at in otype_v for at in agn_types)
    else:
        good_type = True
        agn = False
    return '\n\nSIMBAD\n' + '\n'.join(sim.pformat(max_lines=-1, max_width=-1)[:13]), sim['DISTANCE_RESULT'].min(), good_type, agn
    
def get_ned(coo):
    #NED bug, bug in NED
    if coo.separation(SkyCoord(271.13313251264*u.deg, 67.523408125256*u.deg)).arcsec < 1e-6:
        coo = SkyCoord(271.13313251264*u.deg, 67.5235*u.deg)
    
    def ned_hlp(coo, ts):
        # Recursive pauser to avoid blacklisting
        try:
            ned = Ned.query_region(coo, radius=60*u.arcsec)
        except:
            print('NED blacklisted, pausing for', ts, 's.')
            #NED bug, bug in NED: *** requests.exceptions.ConnectionError: HTTPConnectionPool(host='ned.ipac.caltech.edu', port=80): Read timed out.
            # <SkyCoord (ICRS): (ra, dec) in deg (221.40009467, 68.90898505)>
            # <SkyCoord (ICRS): (ra, dec) in deg (149.59574552, 65.37104883)>
            if ts > 8.1:
                coo = SkyCoord(coo.ra, coo.dec+1e-4*u.deg)
                print('Shifted NED coordinate', 1e-4, 'deg north')
            elif ts > 8.4:
                return None
            time.sleep(ts)
            ned = ned_hlp(coo, ts+0.1)
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
    
def download_data(ii):
    def dl_hlp(cmd):
        retcode = subprocess.call(cmd)
        gtar = glob('GUEST*.tar')
        gftz = glob('*.FTZ')
        file_missing = ' '.join(cmd).split('/')[-1]
        if len(gtar) == 1:
            retcode = subprocess.call(['tar', '-xvf', cwd + gtar[0], '-C', cwd], stderr=subprocess.DEVNULL)
            retcode = subprocess.call(['mv'] + glob(cwd + obs[ii] + '/pps/*') + [cwd + obs[ii]])
            retcode = subprocess.call(['rm', '-rf', cwd + obs[ii] + '/pps'])
            retcode = subprocess.call(['rm', cwd + gtar[0]])
            return True
        elif len(gftz) == 1:
            retcode = subprocess.call(['mkdir', '-p', cwd + obs[ii]])
            retcode = subprocess.call(['mv', cwd + gftz[0], cwd + obs[ii]])
            return True
        elif os.path.isfile(file_missing):
            retcode = subprocess.call(['rm', file_missing])
            return False

        print('ERROR Unknown data format delivered from XSA AIO:', ii)
        print(' '.join(cmd))
        sys.exit(1)

    if os.path.isdir(cwd + obs[ii]): return True
    retcode = subprocess.call(['rm', '-rf', cwd + obs[ii]])
    retcode = subprocess.call(['rm'] + glob('GUEST*.tar'), stderr=subprocess.DEVNULL)
    retcode = subprocess.call(['rm'] + glob('*.FTZ'), stderr=subprocess.DEVNULL)

    tmp = dl_hlp(cmd + [purl.format(obs[ii])])
    return dl_hlp(cmd + [murl.format(obs[ii])]) or tmp




################################################################
# Just read the data
dat = fits.open('/Users/silver/Box Sync/phd/pro/sne/sbo/3xmm_dr8/3XMM_DR8cat_v1.0.fits')[1].data
dat = dat[np.argsort(dat['MJD_START'])]
cwd = '/Users/silver/Desktop/'
os.chdir(cwd)
cmd = ['curl', '-sS', '-O', '-J']
obs_num = '{0:10s}_{1:03X}'
purl = 'http://nxsa.esac.esa.int/nxsa-sl/servlet/data-action-aio?level=PPS&extension=FTZ&name=PIEVLI&obsno={0:10s}'
murl = 'http://nxsa.esac.esa.int/nxsa-sl/servlet/data-action-aio?level=PPS&extension=FTZ&name=MIEVLI&obsno={0:10s}'
# Default from: Users Guide to the XMM-Newton Science Analysis System
# Issue 15.0 Prepared by the XMM-Newton Science Operations Centre Team 30.06.2019
img_size = 1024 # pixel
binning = 80 # pixel, how much to bin the zoomed image (80 is default, results in a pixel size of 4 arcsec)
cts_per_bin = 25 # For the light curves
few_cts = 100 # Reject objects with less than few_cts
det_lim = 10 # Reject objects with lower detection likelihoods
spec_bin = np.logspace(-1,1,20) # keV
plim = 1-0.999999426696856 # 5 sigma
extremely_flaring = 30
very_flaring = 10
flaring = 5 # Fractional excess above the 90th percentile required to qualify as flaring
XMMEA_EP = 65584
XMMEA_EM = 65000
#PN
#>>> list(np.unique(aa))
#[0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29, 32, 36, 40, 44, 48, 56, 64, 68, 72, 80, 84, 88, 65536, 65537, 65540, 65541, 65544, 65545, 65552, 65553, 65568, 65572, 65576, 65584, 524297, 524301, 524313, 2097152, 2097156, 2097160, 2097164, 2097168, 2097176]
#>>> list(np.unique(bb))
#[0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29, 32, 36, 40, 44, 48, 56, 64, 68, 72, 80, 84, 88, 65536, 65537, 65540, 65541, 65544, 65545, 65552, 65553, 65568, 65572, 65576, 65584]
#
#MOS
#>>> list(np.unique(aa))
#[0, 1, 2, 32, 64, 65, 66, 256, 258, 320, 2048, 2050, 2112, 2114, 2304, 65536, 65537, 65538, 65539, 65600, 65602, 65792, 65794, 65824, 67584, 67586, 67648, 67840, 4194304, 4194305, 4194306, 4194368, 4194560, 4194624, 4196352, 4196608, 4259840, 4259842, 4259904, 4260096, 4261888]
#>>> list(np.unique(bb))
#[0, 1, 2, 32, 64, 66, 256, 258, 320, 2048, 2050, 2112, 2304]

#print(dat.dtype.names)

detid = dat['DETID']
srcid = dat['SRCID']
iauid = dat['IAUNAME']
num = dat['SRC_NUM']
obs = dat['OBS_ID']
t0 = dat['MJD_START']
t1 = dat['MJD_STOP']
ra = dat['RA']

#NED bug, bug in NED
ra = np.where(ra < 0.0015, 0.001502, ra) # Problems when quering NED with RA that formats into scientific format rather than 0.0001 (i.e. arcsec less than 0.0001 or hour angles less than 0.0015). 0.001501 doesn't work, need at least 0.001502?

de = dat['DEC']
coo = SkyCoord(ra*u.deg, de*u.deg, frame='icrs')
err = dat['POSERR']
ll = dat['LII']
bb = dat['BII']
flag_sum = dat['SUM_FLAG']
first = dat['MJD_FIRST']
last = dat['MJD_LAST']
ndet = dat['N_DETECTIONS']
ext = dat['SC_EXTENT']
cts = dat['EP_8_CTS']
cts_err = dat['EP_8_CTS_ERR']
rate = dat['EP_8_RATE']
rate_err = dat['EP_8_RATE_ERR']
flux = dat['EP_8_FLUX']
flux_err = dat['EP_8_FLUX_ERR']
det_ml = dat['EP_8_Det_ML']

################################################################
for arg in sys.argv[1:]:    
    ii = np.where((obs == arg.split('_')[0]) & (num == int(arg.split('_')[1], 16)))[0][0]
    xoi = Xoi(ii)
    download_data(ii)
    observation = Observation(ii)
    xoi.ext_evt(observation.wcs)
    xoi.mk_img(observation)
    xoi.mk_lc()
    xoi.mk_spec()
    xoi.mk_postcard()
    # retcode = subprocess.call(['rm', '-rf', cwd + obs[ii]])
    del(observation)
# db()
