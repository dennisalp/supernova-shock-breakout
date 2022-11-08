'''
2021-05-21, Dennis Alp, dalp@kth.se

Scan all pointed ROSAT PSPC data. Search for transients, pulsars, and other stuff.

'''

from __future__ import division, print_function
import os
from pdb import set_trace as db
import sys
import time
from glob import glob
from datetime import date
import subprocess

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import scipy.stats as sts
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.wcs import WCS
from astropy.io import fits
from astropy import units
from astroquery.vizier import Vizier
from astroquery.simbad import Simbad
from astropy.time import Time

# For the catalogs
gaia_col = ['_r', 'RA_ICRS', 'DE_ICRS', 'Plx', 'e_Plx', 'Gmag', 'e_Gmag']
mass_col = ['_r', 'RAJ2000', 'DEJ2000', 'Hmag', 'e_Hmag'] 
sdss_col = ['_r', 'RA_ICRS', 'DE_ICRS', 'class', 'rmag', 'e_rmag', 'zsp', 'zph', 'e_zph', '__zph_']
Simbad._VOTABLE_FIELDS = ['distance_result', 'ra', 'dec', 'plx', 'plx_error','rv_value', 'z_value', 'otype(V)']



################################################################
#
def pi2kev(pi):
    return pi*0.01
def kev2pi(kev):
    return kev/0.01

def get_wcs(ff):
    dd = fits.open(ff)
    wcs = WCS()
    wcs.wcs.crpix = [dd[2].header['TCRPX1'], dd[2].header['TCRPX2']]
    wcs.wcs.cdelt = [dd[2].header['TCDLT1'], dd[2].header['TCDLT2']]
    wcs.wcs.crval = [dd[2].header['TCRVL1'], dd[2].header['TCRVL2']]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return wcs

class Observation:
    def __init__(self):
        print('Reading data')
        self.tt = np.empty(0)
        self.ee = np.empty(0)
        self.xx = np.empty(0)
        self.yy = np.empty(0)
        
        ff = obs + '_bc.fits'
        dd = fits.open(ff)[2].data
        print('Standard events ', dd['TIME'].size)
        print('Rejected events ', fits.open(ff)[3].data['TIME'].size)


        good = (dd['PI'] >= kev2pi(ene_low)) & (dd['PI'] <= kev2pi(ene_high))
            
        self.tt = dd['TIME'][good]
        self.t2 = self.tt.copy()
        dt = np.diff(self.tt)
        for tt in np.where(dt>100)[0]:
            self.t2[tt+1:] = self.t2[tt+1:]-dt[tt]
        
        self.ee = dd['PI'][good]
        self.xx = dd['X'][good]
        self.yy = dd['Y'][good]

        self.wcs = get_wcs(ff)
        self.pix_size = np.abs(self.wcs.pixel_scale_matrix[0,0]*3600)
        
        self.lcb = np.arange(self.tt[0], self.tt[-1], lc_bin)
        self.lc, _ = np.histogram(self.tt, self.lcb)
        self.lc = self.lc/np.diff(self.lcb)
        
        xbin = np.arange(self.xx.min(), self.xx.max()+1e-3, binning)
        ybin = np.arange(self.yy.min(), self.yy.max()+1e-3, binning)
        self.img, _, _ = np.histogram2d(self.xx, self.yy, [xbin, ybin])

        
        # V2 # Light curve
        # self.out = '{0:s}_obs_lc'.format(self.oi)
        # ouf_name = ouf + self.out + '.pdf'
        # oud_name = oud + self.out
        # fig = plt.figure(figsize=(5, 3.75))

        # tmp = np.empty(2*self.lcb.size-2)
        # tmp[:-1:2] = self.lcb[:-1]
        # tmp[1::2] = self.lcb[1:]
        # shift = int(tmp[0]//1000*1000)
        # tmp = tmp - shift
        # plt.plot(tmp, self.lc.repeat(2), color='k')
        # plt.xlabel('Time-{0:d} (s)'.format(shift))
        # plt.ylabel('Rate (cts/s)')
        # plt.ylim(bottom=0)
        # plt.title(self.oi + ' ' + self.target)
        # plt.savefig(ouf_name, bbox_inches='tight', pad_inches=0.1, dpi=300)
        # plt.close()
        # np.save(oud_name, np.c_[self.lcb[:-1], self.lc])
        




        
    
        




class Transient:
    def __init__(self, obs, obj):
        self.tb = obj['t']
        self.x0 = obj['x']
        self.y0 = obj['y']
        self.sky = obj['sky']
        xx, yy = np.array(obs.wcs.all_world2pix(self.sky.ra, self.sky.dec, 1))
        self.coo = np.array([xx, yy])        
        self.stat = obj['stat']
        self.oi = obs.oi
        self.pix_size = obs.pix_size
        
        # Query catalogs
        viz = get_viz(self.sky)
        sim = get_sim(self.sky)

        # Parse meta data
        self.target = target
        buf = 100*'#' + '\nIndex ' + str(obs.ii) + '\n' + 100*'#' + '\n\n'
        buf = buf + 'Obs. Date: ' + obs.epoch.iso + '\n'
        buf = buf + 'Obs. ID: ' + obs.oi + ', Target: ' + obs.target + '\n'
        buf = buf + 'RA, Dec:' + '{0:>16s}{1:>16s}'.format(*self.sky.to_string('hmsdms').split(' '))+ '\n'
        buf = buf + 'RA, Dec:' + '{0:>16s}{1:>16s}'.format(*self.sky.to_string('decimal').split(' ')) + '\n'
        buf = buf + 'l, b:   ' + '{0:>16s}{1:>16s}'.format(*self.sky.galactic.to_string('decimal').split(' ')) + '\n'
        astrometry = np.sqrt(self.x0[1]**2+self.y0[1]**2)
        if astrometry < 60:
            buf = buf + 'Astrometric accuracy: {0:7.3f} arcsec\n'.format(astrometry)
        else:
            buf = buf + 'Astrometric accuracy undetermined; fitting failed\n'
        buf = buf + 'Exposure: {0:3.0f} ks, Detection statistic: {1:6.1f}'.format((obs.t2.max()-obs.t2.min())/1.e3, self.stat) + '\n'

        self.viz = viz
        self.sim = sim
        self.buf = buf
        self.src_rad = 60/self.pix_size
        self.bkg_rad = [2*self.src_rad, 3*self.src_rad]
        self.backscal = self.src_rad**2/(self.bkg_rad[1]**2-self.bkg_rad[0]**2)

    def ext_evt(self, obs):
        def get_src_bkg_idx(self, tt, xx, yy): # For light curves and spectra
            rad = (xx-self.coo[0])**2+(yy-self.coo[1])**2
            gd =  (tt > self.tb[0]) & (tt < self.tb[3])
            src = (rad < self.src_rad**2) & gd
            bkg = (rad > self.bkg_rad[0]**2) & (rad < self.bkg_rad[1]**2) & gd
            return src, bkg

        def get_img_idx(self, tt, xx, yy): # For images
            idx = (tt > self.tb[1]) & (tt < self.tb[2])
            idx = idx & (xx > self.coo[0]-self.bkg_rad[1]) & (xx < self.coo[0]+self.bkg_rad[1]+1e-3)
            idx = idx & (yy > self.coo[1]-self.bkg_rad[1]) & (yy < self.coo[1]+self.bkg_rad[1]+1e-3)
            return idx

        # For light curves and spectra
        src, bkg = get_src_bkg_idx(self, obs.t2, obs.xx, obs.yy)
        self.src_tt = obs.t2[src]
        self.src_pi = obs.ee[src]
        self.bkg_tt = obs.t2[bkg]
        self.bkg_pi = obs.ee[bkg]

        order = np.argsort(self.src_tt)
        self.src_tt = self.src_tt[order]
        self.src_pi = self.src_pi[order]
        order = np.argsort(self.bkg_tt)
        self.bkg_tt = self.bkg_tt[order]
        self.bkg_pi = self.bkg_pi[order]
        
        # For images
        idx = get_img_idx(self, obs.t2, obs.xx, obs.yy)
        self.xx = obs.xx[idx]
        self.yy = obs.yy[idx]

    def mk_lc(self):
        self.tlc = np.linspace(self.tb[0], self.tb[3], lc_nbin+1)
        self.src_lc, _ = np.histogram(self.src_tt, self.tlc)
        self.src_lc = self.src_lc/np.diff(self.tlc)
        self.bkg_lc, _ = np.histogram(self.bkg_tt, self.tlc)
        self.bkg_lc = self.backscal*self.bkg_lc/np.diff(self.tlc)
        self.sub_lc = self.src_lc - self.bkg_lc
        




################################################################
# Just read the data
cwd = '/Users/silver/Desktop/'
os.chdir(cwd)
cts_per_bin = 8 # For the light curves


################################################################
obs = sys.argv[1].split('_')[0]
my_id = sys.argv[1].split('_')[1:]
coo = SkyCoord(float(sys.argv[2])*u.deg, float(sys.argv[3])*u.deg)
src = Source()
observation = Observation(obs)
src.ext_evt(observation.wcs)
src.mk_lc()
