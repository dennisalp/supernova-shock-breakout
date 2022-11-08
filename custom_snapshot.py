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
from astropy.time import Time


################################################################
#
class Observation:
    def __init__(self, ii):
        self.xx = [np.empty(0), np.empty(0), np.empty(0)]
        self.yy = [np.empty(0), np.empty(0), np.empty(0)]
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
            good = good & (dd['TIME'] > t0) & (dd['TIME'] < t1)
            good = good & (dd['X'] > 0.) # XMM puts some invalid photons at (-99999999.0, -99999999.0)
            good = good & (dd['Y'] > 0.)

            if self.cam == 'PN': tmp = 0
            elif self.cam == 'M1': tmp = 1
            elif self.cam == 'M2': tmp = 2
            
            self.xx[tmp] = np.concatenate((self.xx[tmp], dd['X'][good]))
            self.yy[tmp] = np.concatenate((self.yy[tmp], dd['Y'][good]))

        self.wcs = get_wcs(ff)


        
class Xoi:
    def __init__(self, ii):
        self.ii = ii
        if ext[ii] >= 1.e-5:
            self.src_rad = np.max((400, ext[ii]/0.05)) # pixels, pixel size of 50 mas
            self.bkg_rad = [np.max((1000, self.src_rad)), np.max((2000, 2*self.src_rad))] # "To avoid non-converging fitting an upper limit of 80" is imposed."
        else:
            self.src_rad = 400 # pixels, pixel size of 50 mas
            self.bkg_rad = [1000, 2000]
        
    def ext_evt(self, wcs):
        xx, yy = np.array(wcs.all_world2pix(coo[self.ii].ra, coo[self.ii].dec, 1))
        self.coo = np.array([xx, yy])

    def mk_img(self, observation):
        xbin = np.arange(self.coo[0]-self.bkg_rad[1], self.coo[0]+self.bkg_rad[1]+1e-3, binning)
        ybin = np.arange(self.coo[1]-self.bkg_rad[1], self.coo[1]+self.bkg_rad[1]+1e-3, binning)
        self.img = 3*[None]
        for ii in range(3):
            self.img[ii], _, _ = np.histogram2d(observation.xx[ii], observation.yy[ii], [xbin, ybin])
#        plt.imshow(gaussian_filter(np.log10(observation.img+1),2).T, vmin=0, vmax=np.percentile(gaussian_filter(np.log10(observation.img+1), 2), 99.9), cmap='afmhot', origin='lower')
    def mk_postcard(self):
        fig = plt.figure(figsize=(12, 4))
#        gs.GridSpec(2,3)

        for ii in range(6):
            ax = plt.subplot2grid((2,3), (ii//3, np.mod(ii,3)), colspan=1, rowspan=1)
            tmp = self.img[np.mod(ii,3)].T
            tmp = gaussian_filter(tmp, 2*(ii//3))
#            tmp = np.log10(tmp)
            tmp2 = np.percentile(tmp,99.999)
            tmp = np.where(tmp > tmp2, tmp2, tmp)
            plt.imshow(tmp, cmap='afmhot', origin='lower')
            tmp = plt.Circle((self.bkg_rad[1]/binning-0.5, self.bkg_rad[1]/binning-0.5), self.src_rad/binning, color='w', lw=2, fill=False)
            ax.add_artist(tmp)
            tmp = plt.Circle((self.bkg_rad[1]/binning-0.5, self.bkg_rad[1]/binning-0.5), self.bkg_rad[0]/binning, color='w', lw=1, fill=False)
            ax.add_artist(tmp)
            tmp = plt.Circle((self.bkg_rad[1]/binning-0.5, self.bkg_rad[1]/binning-0.5), self.bkg_rad[1]/binning, color='w', lw=1, fill=False)
            ax.add_artist(tmp)
            ax.set_axis_off()

        plt.show()
        return 1
   
def get_wcs(ff):
    dd = fits.open(ff)
    wcs = WCS()
    wcs.wcs.crpix = [dd[0].header['REFXCRPX'], dd[0].header['REFYCRPX']]
    wcs.wcs.cdelt = [dd[0].header['REFXCDLT'], dd[0].header['REFYCDLT']]
    wcs.wcs.crval = [dd[0].header['REFXCRVL'], dd[0].header['REFYCRVL']]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return wcs
    
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
img_size = 1024 # pixel
binning = 80 # pixel, how much to bin the zoomed image (80 is default, results in a pixel size of 4 arcsec)
XMMEA_EP = 65584
XMMEA_EM = 65000

num = dat['SRC_NUM']
obs = dat['OBS_ID']
ra = dat['RA']

#NED bug, bug in NED
ra = np.where(ra < 0.0015, 0.001502, ra) # Problems when quering NED with RA that formats into scientific format rather than 0.0001 (i.e. arcsec less than 0.0001 or hour angles less than 0.0015). 0.001501 doesn't work, need at least 0.001502?
de = dat['DEC']
coo = SkyCoord(ra*u.deg, de*u.deg, frame='icrs')
ext = dat['SC_EXTENT']

################################################################
arg = sys.argv[1]
tmp = float(sys.argv[2])
t0 = tmp+float(sys.argv[3])
t1 = t0+float(sys.argv[4])
ii = np.where((obs == arg.split('_')[0]) & (num == int(arg.split('_')[1], 16)))[0][0]
xoi = Xoi(ii)
download_data(ii)
observation = Observation(ii)
xoi.ext_evt(observation.wcs)
xoi.mk_img(observation)
xoi.mk_postcard()
# retcode = subprocess.call(['rm', '-rf', cwd + obs[ii]])
# db()
