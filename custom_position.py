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
    def __init__(self):
        self.xx = np.empty(0)
        self.yy = np.empty(0)
        for ff in glob(cwd + obs + '/*FTZ'):
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
                print('ERROR Unknown instrument:', cam, obs)
                sys.exit(1)
                
            good = good & (dd['PI'] > 300.) & (dd['PI'] < 2000.)
#            good = good & (dd['X'] > -99999998) # XMM puts some invalid photons at (-99999999.0, -99999999.0)
#            good = good & (dd['Y'] > -99999998)
            good = good & (dd['X'] > 0.) # XMM puts some invalid photons at (-99999999.0, -99999999.0)
            good = good & (dd['Y'] > 0.)
            self.xx = np.concatenate((self.xx, dd['X'][good]))
            self.yy = np.concatenate((self.yy, dd['Y'][good]))

        self.wcs = get_wcs(ff)


        
class Xoi:
    def __init__(self):
        self.src_tt = np.empty(0)
        self.src_pi = np.empty(0)
        self.bkg_tt = np.empty(0)
        self.bkg_pi = np.empty(0)
        self.src_rad = 400
        self.bkg_rad = [800, 1200]
        self.backscal = self.src_rad**2/(self.bkg_rad[1]**2-self.bkg_rad[0]**2)

    def ext_evt(self, wcs):
        for ff in glob(cwd + obs + '/*FTZ'):
            self.cam = ff.split('/')[-1][11:13]
            xx, yy = np.array(wcs.all_world2pix(coo.ra, coo.dec, 1))
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

    def mk_lc(self):
        self.tlc = np.arange(self.src_tt[0], self.src_tt[-1], float(sys.argv[4]))
        self.src_lc, _ = np.histogram(self.src_tt, self.tlc)
        self.src_lc = self.src_lc/np.diff(self.tlc)
        self.bkg_lc, _ = np.histogram(self.bkg_tt, self.tlc)
        self.bkg_lc = self.backscal*self.bkg_lc/np.diff(self.tlc)
        self.sub_lc = self.src_lc - self.bkg_lc
        tmp = np.empty(2*self.tlc.size-2)
        tmp[:-1:2] = self.tlc[:-1]
        tmp[1::2] = self.tlc[1:]
        shift = int(tmp[0]//1000*1000)
        tmp = tmp - shift

        fig = plt.figure(figsize=(5, 3.75))
        ax = plt.gca()
        plt.plot(tmp, self.src_lc.repeat(2), label='Source', color='greenyellow')
        plt.plot(tmp, self.bkg_lc.repeat(2), label='Background', color='gold')
        plt.plot(tmp, self.sub_lc.repeat(2), label='Subtracted', color='k')
        ax.set_title(obs + '_' + my_id)
        ax.legend()
        ax.set_xlabel('Time-{0:d} (s)'.format(shift))
        ax.set_ylabel('Rate (cts/s)')
        ax.set_ylim(bottom=0)
        plt.show()
   
def get_wcs(ff):
    dd = fits.open(ff)
    wcs = WCS()
    wcs.wcs.crpix = [dd[0].header['REFXCRPX'], dd[0].header['REFYCRPX']]
    wcs.wcs.cdelt = [dd[0].header['REFXCDLT'], dd[0].header['REFYCDLT']]
    wcs.wcs.crval = [dd[0].header['REFXCRVL'], dd[0].header['REFYCRVL']]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return wcs

    
def download_data():
    def dl_hlp(cmd):
        retcode = subprocess.call(cmd)
        gtar = glob('GUEST*.tar')
        gftz = glob('*.FTZ')
        file_missing = ' '.join(cmd).split('/')[-1]
        if len(gtar) == 1:
            retcode = subprocess.call(['tar', '-xvf', cwd + gtar[0], '-C', cwd], stderr=subprocess.DEVNULL)
            retcode = subprocess.call(['mv'] + glob(cwd + obs + '/pps/*') + [cwd + obs])
            retcode = subprocess.call(['rm', '-rf', cwd + obs + '/pps'])
            retcode = subprocess.call(['rm', cwd + gtar[0]])
            return True
        elif len(gftz) == 1:
            retcode = subprocess.call(['mkdir', '-p', cwd + obs])
            retcode = subprocess.call(['mv', cwd + gftz[0], cwd + obs])
            return True
        elif os.path.isfile(file_missing):
            retcode = subprocess.call(['rm', file_missing])
            return False

        print('ERROR Unknown data format delivered from XSA AIO:', obs)
        print(' '.join(cmd))
        sys.exit(1)

    if os.path.isdir(cwd + obs): return True
    retcode = subprocess.call(['rm', '-rf', cwd + obs])
    retcode = subprocess.call(['rm'] + glob('GUEST*.tar'), stderr=subprocess.DEVNULL)
    retcode = subprocess.call(['rm'] + glob('*.FTZ'), stderr=subprocess.DEVNULL)

    tmp = dl_hlp(cmd + [purl.format(obs)])
    return dl_hlp(cmd + [murl.format(obs)]) or tmp




################################################################
cwd = '/Users/silver/Desktop/'
os.chdir(cwd)
cts_per_bin = 25 # For the light curves
cmd = ['curl', '-sS', '-O', '-J']
purl = 'http://nxsa.esac.esa.int/nxsa-sl/servlet/data-action-aio?level=PPS&extension=FTZ&name=PIEVLI&obsno={0:10s}'
murl = 'http://nxsa.esac.esa.int/nxsa-sl/servlet/data-action-aio?level=PPS&extension=FTZ&name=MIEVLI&obsno={0:10s}'
XMMEA_EP = 65584
XMMEA_EM = 65000


################################################################
obs = sys.argv[1].split('_')[0]
my_id = sys.argv[1].split('_')[1]
coo = SkyCoord(float(sys.argv[2])*u.deg, float(sys.argv[3])*u.deg)
xoi = Xoi()
download_data()
observation = Observation()
xoi.ext_evt(observation.wcs)
xoi.mk_lc()
