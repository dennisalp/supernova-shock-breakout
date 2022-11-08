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
    def __init__(self, ii):
        print('Reading data')
        self.ii = ii
        self.oi = obs[ii]
        self.target = target[ii]
        self.epoch = epoch[ii]
        self.tt = np.empty(0)
        self.ee = np.empty(0)
        self.xx = np.empty(0)
        self.yy = np.empty(0)
        
        ff = self.oi + '_bc.fits'
        dd = fits.open(ff)[2].data
        print('Standard events ', dd['TIME'].size)
        print('Rejected events ', fits.open(ff)[3].data['TIME'].size)


        good = (dd['PI'] >= kev2pi(ene_low)) & (dd['PI'] <= kev2pi(ene_high))
        # good = good & (dd['X'] > 0.) # NuSTAR puts some invalid photons at (-1, -1)
        # good = good & (dd['Y'] > 0.)
        # good = good & (dd['X'] < 100000.+dd['X'][good].mean())
        # good = good & (dd['Y'] < 100000.+dd['Y'][good].mean())
            
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
        
        # # Zoomed image
        # self.out = '{0:s}_obs_img_lin'.format(self.oi)
        # ouf_name = ouf + self.out + '.pdf'
        # fig = plt.figure(figsize=(5, 3.75))
        # plt.imshow(self.img.T, cmap='afmhot', origin='lower')
        # plt.title(self.oi + ' ' + self.target)
        # plt.gca().set_axis_off()
        # plt.savefig(ouf_name, bbox_inches='tight', pad_inches=0.1, dpi=300)
        # plt.close()

        # self.out = '{0:s}_obs_img_log'.format(self.oi)
        # ouf_name = ouf + self.out + '.pdf'
        # fig = plt.figure(figsize=(5, 3.75))
        # plt.imshow(np.log10(self.img.T+1), cmap='afmhot', origin='lower')
        # plt.title(self.oi + ' ' + self.target)
        # plt.gca().set_axis_off()
        # plt.savefig(ouf_name, bbox_inches='tight', pad_inches=0.1, dpi=300)
        # plt.close()




        

    def find_transients(self):
        def get_shifts(ti):
            return [[      0,       0,            0],
                    [sbin//2,       0,            0],
                    [      0, sbin//2,            0],
                    [sbin//2, sbin//2,            0],
                    [      0,       0, tbins[ti]//2],
                    [sbin//2,       0, tbins[ti]//2],
                    [      0, sbin//2, tbins[ti]//2],
                    [sbin//2, sbin//2, tbins[ti]//2]]

        def mk_cubes(self, ti):
            cubes = np.empty((self.xbin.size-1, self.ybin.size-1, self.tbin[ti].size-1, 8))
            shifts = get_shifts(ti)
            
            for jj, sh in enumerate(shifts):
                bins = [self.xbin+sh[0], self.ybin+sh[1], self.tbin[ti]+sh[2]]
                cubes[:,:,:,jj], _ = np.histogramdd(np.c_[self.xx, self.yy, self.t2], bins=bins)

            return cubes

        def get_likelihood(cubes):
            # Current cell
            obs = cubes[1:-1, 1:-1, 1:-1, :]
            # obs = np.where(obs < 20, 0, obs)

            # Before
            mu = 0.5*(cubes[1:-1, 1:-1,  :-2, :] - 0.25*(cubes[ :-2, 1:-1,  :-2, :]+
                                                         cubes[2:  , 1:-1,  :-2, :]+
                                                         cubes[1:-1,  :-2,  :-2, :]+
                                                         cubes[1:-1, 2:  ,  :-2, :]))
            # After
            mu = mu + 0.5*(cubes[1:-1, 1:-1, 2:  , :] - 0.25*(cubes[ :-2, 1:-1, 2:  , :]+
                                                              cubes[2:  , 1:-1, 2:  , :]+
                                                              cubes[1:-1,  :-2, 2:  , :]+
                                                              cubes[1:-1, 2:  , 2:  , :]))
            # Shift all values <= 0 to 0.125
            mu = np.where(mu < 0.124, 0.125, mu)
            # Spatial background
            mu = mu + 0.25*(cubes[ :-2, 1:-1, 1:-1, :]+
                            cubes[2:  , 1:-1, 1:-1, :]+
                            cubes[1:-1,  :-2, 1:-1, :]+
                            cubes[1:-1, 2:  , 1:-1, :])

            obs = sts.poisson.sf(obs, mu)
            obs = np.where(obs < 1.e-99, 1e-99, obs)
            return -np.log(obs), np.average(cubes[1:-1,1:-1,:,:], axis=2)

        def get_test_stat(cubes, cts):
            tbin_nr = np.argmax(cubes, axis=2) # Get index for time of the transient
            cubes = cubes.max(axis=2) # Collapse time
            cubes = np.where(cts < 1e-4, 0., cubes)
            cubes = np.where(np.abs(cubes+np.log(sts.poisson.sf(0, 0.125))) < 1e-12, 0., cubes)
            shift_nr = np.argmax(cubes, axis=2) # Get index for best shift
            idx = cubes.argmax(axis=2) # Advanced indexing to get the index in time_nr
            mm, nn = idx.shape
            mm, nn = np.ogrid[:mm,:nn]
            tbin_nr = tbin_nr[mm, nn, idx] # Index in tbin for best shift
            cubes = cubes.max(axis=2) # Keep only best shift
            return cubes, tbin_nr, shift_nr
        
        def get_obj(self, imgs, tbin_nr, shift_nr):
            def help_gauss(dummy, x0, y0, aa, sig, bkg):
                yy, xx = np.ogrid[:nn,:nn]
                xx = xx-x0
                yy = yy-y0
                rr = xx**2+yy**2
                global gau
                gau = aa*np.exp(-rr/sig**2)+bkg
                return np.reshape(gau, nn*nn)

            idx = imgs.argmax(axis=2)
            mm, nn = idx.shape
            mm, nn = np.ogrid[:mm,:nn]
            best_tbin = tbin_nr[mm, nn, idx]
            best_shift = shift_nr[mm, nn, idx]
            img = imgs.max(axis=2)
            tmp, _ = np.histogram(np.where(img.ravel() > 999., 999.5, img.ravel()), stat_bin)
            np.save(oud + self.oi + '_xrt_stat', tmp.astype(np.int))
            
            # Find peaks
            xx, yy = np.unravel_index(np.argsort(img.ravel())[::-1], img.shape)
            obj = []
            for jj in range(0, xx.size):
                val = img[xx[jj], yy[jj]]
                if val < -1.e98: # I use this to kill neighbors
                    continue
                elif val < test_stat_lim:
                    break

                ti = idx[xx[jj], yy[jj]] # Index of timescale in tbins array
                tbnr = best_tbin[xx[jj], yy[jj]] # Index of bin in temporal dimension
                shnr = best_shift[xx[jj], yy[jj]] # Index for the shifts
                shift = get_shifts(ti)[shnr]
                xb = self.xbin[xx[jj]+1:xx[jj]+3]+shift[0] # Shift by 1 because of the convolution
                yb = self.ybin[yy[jj]+1:yy[jj]+3]+shift[1]
                tb = self.tbin[ti][tbnr:tbnr+4]+shift[2] # Extra time bin to see before and after
                obj.append({'x': xb, 'y': yb, 't': tb, 'stat': val})

                img[np.max((0, xx[jj]-1)):xx[jj]+2, np.max((0, yy[jj]-1)):yy[jj]+2] = -1.e99
                

            
            # Get raw data and fit the position
            nn = 10
            for oi, oo in enumerate(obj):
                gd = (self.xx < oo['x'][1]) & (self.xx > oo['x'][0])
                gd = gd & (self.yy < oo['y'][1]) & (self.yy > oo['y'][0])
                gd = gd & (self.t2 < oo['t'][2]) & (self.t2 > oo['t'][1])
                xb = np.linspace(oo['x'][0], oo['x'][1], nn+1)
                yb = np.linspace(oo['y'][0], oo['y'][1], nn+1)
                im, _, _ = np.histogram2d(self.xx[gd], self.yy[gd], bins=(xb, yb))
                

                ang_res = 20/(self.pix_size*sbin/nn) # Angular resolution of NuSTAR in current pixel units
                guess = np.array([nn/2, nn/2, im.max(), ang_res, np.percentile(im, 20)])
                bounds = (np.array([0, 0, 0, 0.3*ang_res, 0]), np.array([nn, nn, 1.4*im.max(), 3*ang_res, np.percentile(im, 90)+0.01]))
                try:
                    pars, covar = curve_fit(help_gauss, np.arange(0, nn*nn), np.reshape(im, nn*nn), guess, bounds=bounds)
                    err = np.sqrt(np.diag(covar))
                except:
                    pars = guess
                    err = 99999*np.ones(pars.size)

                # Note the reversal of x and y here to transpose
                # the coordinate from Python (i.e. matrix standard) to image standard
                # Shift by 0.5 because of matrix convention of indexing in the middle of the pixel,
                # which shiftes the index by half with respect to the bins.
                oo['x'] = [np.interp(pars[1]+0.5, np.arange(xb.size), xb), err[0]*(50*sbin/nn)/self.pix_size]
                oo['y'] = [np.interp(pars[0]+0.5, np.arange(yb.size), yb), err[1]*(50*sbin/nn)/self.pix_size]

                ra,de=self.wcs.wcs_pix2world(oo['x'][0], oo['y'][0], 1)
                oo['sky'] = SkyCoord(ra*u.deg, de*u.deg)

            return obj



        ################################################################
        print('Finding transients')
        self.xbin = np.arange(self.xx.min(), self.xx.max(), sbin)
        self.ybin = np.arange(self.yy.min(), self.yy.max(), sbin)
        self.tbin = []
        for tbin in tbins:
            tmp = np.arange(self.t2.min(), self.t2.max(), tbin)
            if len(tmp) <= 3:
                break
            self.tbin.append(tmp)

        imgs = np.empty((self.xbin.size-3, self.ybin.size-3, len(self.tbin)))
        tbin_nr = np.empty((self.xbin.size-3, self.ybin.size-3, len(self.tbin))).astype('int64')
        shift_nr = np.empty((self.xbin.size-3, self.ybin.size-3, len(self.tbin))).astype('int64')
        for ti in range(len(self.tbin)):
            cubes = mk_cubes(self, ti)
            cubes, cts = get_likelihood(cubes)
            imgs[:,:,ti], tbin_nr[:,:,ti], shift_nr[:,:,ti] = get_test_stat(cubes, cts)
            
        del(cubes)
        self.obj = get_obj(self, imgs, tbin_nr, shift_nr)

    def gen_products(self):
        print('{0:d} transient(s), generating products'.format(len(self.obj)))
        oi = -1
        for oi, obj in enumerate(self.obj):
            src = Transient(self, obj)
            src.ext_evt(self)
            src.mk_img()
            src.mk_lc()
            src.mk_spec()
            src.get_hips()
            src.mk_postcard(oi)
        print('Generated {0:d} transient product(s)'.format(oi+1))






        
    def find_sources(self):
        def help_gauss(dummy, x0, y0, aa, sig, bkg):
            yy, xx = np.ogrid[:sbin,:sbin]
            xx = xx-x0
            yy = yy-y0
            rr = xx**2+yy**2
            global gau
            gau = aa*np.exp(-rr/sig**2)+bkg
            return np.reshape(gau, sbin*sbin)
        
        img, _, _ = np.histogram2d(self.xx, self.yy, bins=[self.xbin, self.ybin])
        flux_lim = np.median(img[img>0.1])
        flux_std = np.sqrt(flux_lim)
        flux_lim = flux_lim+10*flux_std
        flux_lim = FLUX_LIM if FLUX_LIM > flux_lim+10*flux_std else flux_lim+10*flux_std
        
        # Find peaks
        img = gaussian_filter(img, 1)
        xx, yy = np.unravel_index(np.argsort(img.ravel())[::-1], img.shape)
        obj = []
        for jj in range(0, xx.size):
            val = img[xx[jj], yy[jj]]
            if val < 0.999999*img[np.max((0, xx[jj]-1)):xx[jj]+2, np.max((0, yy[jj]-1)):yy[jj]+2].max():
                continue
            elif val < flux_lim:
                break

            obj.append({'x': self.xbin[xx[jj]:xx[jj]+2], 'y': self.ybin[yy[jj]:yy[jj]+2], 'stat': val})

        # Get raw data and fit the position
        for oi, oo in enumerate(obj):
            gd = (self.xx < oo['x'][1]) & (self.xx > oo['x'][0])
            gd = gd & (self.yy < oo['y'][1]) & (self.yy > oo['y'][0])
            xb = np.linspace(oo['x'][0], oo['x'][1], sbin+1)
            yb = np.linspace(oo['y'][0], oo['y'][1], sbin+1)
            im, _, _ = np.histogram2d(self.xx[gd], self.yy[gd], bins=(xb, yb))
            

            ang_res = 20/(self.pix_size*sbin/sbin) # Angular resolution of NuSTAR in current pixel units
            guess = np.array([sbin/2, sbin/2, im.max(), ang_res, np.percentile(im, 20)])
            bounds = (np.array([0, 0, 0, 0.3*ang_res, 0]), np.array([sbin, sbin, 1.4*im.max(), 3*ang_res, np.percentile(im, 90)+0.01]))
            try:
                pars, covar = curve_fit(help_gauss, np.arange(0, sbin*sbin), np.reshape(im, sbin*sbin), guess, bounds=bounds)
                err = np.sqrt(np.diag(covar))
            except:
                pars = guess
                err = 99999*np.ones(pars.size)
                
            # Note the reversal of x and y here to transpose
            # the coordinate from Python (i.e. matrix standard) to image standard
            # Shift by 0.5 because of matrix convention of indexing in the middle of the pixel,
            # which shiftes the index by half with respect to the bins.
            oo['x'] = [np.interp(pars[1]+0.5, np.arange(xb.size), xb), err[0]*self.pix_size]
            oo['y'] = [np.interp(pars[0]+0.5, np.arange(yb.size), yb), err[1]*self.pix_size]

            ra,de=self.wcs.wcs_pix2world(oo['x'][0], oo['y'][0], 1)
            oo['sky'] = SkyCoord(ra*u.deg, de*u.deg)

        print('{0:d} source(s), generating products'.format(len(obj)))
        oi = -1
        for oi, obj in enumerate(obj):
            src = Source(self, obj)
            src.ext_evt(self)
            src.mk_img()
            src.mk_lc()
            src.mk_pds()
            src.mk_spec()
            src.get_hips()
            src.mk_postcard(oi)
        print('Generated {0:d} source product(s)'.format(oi+1))







    
        




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
        
    def mk_img(self):
        xbin = np.arange(self.coo[0]-self.bkg_rad[1], self.coo[0]+self.bkg_rad[1]+1e-3, binning)
        ybin = np.arange(self.coo[1]-self.bkg_rad[1], self.coo[1]+self.bkg_rad[1]+1e-3, binning)
        self.img, _, _ = np.histogram2d(self.xx, self.yy, [xbin, ybin])

    def mk_lc(self):
        self.tlc = np.linspace(self.tb[0], self.tb[3], lc_nbin+1)
        self.src_lc, _ = np.histogram(self.src_tt, self.tlc)
        self.src_lc = self.src_lc/np.diff(self.tlc)
        self.bkg_lc, _ = np.histogram(self.bkg_tt, self.tlc)
        self.bkg_lc = self.backscal*self.bkg_lc/np.diff(self.tlc)
        self.sub_lc = self.src_lc - self.bkg_lc
        
    def mk_spec(self):
        self.src_spec, _ = np.histogram(pi2kev(self.src_pi), spec_bin)
        self.bkg_spec, _ = np.histogram(pi2kev(self.bkg_pi), spec_bin)
        self.src_spec = self.src_spec/np.diff(spec_bin)
        self.bkg_spec = self.backscal*self.bkg_spec/np.diff(spec_bin)

    def get_hips(self):
        # https://aladin.u-strasbg.fr/hips/list
        url = hips.format('CDS/P/PanSTARRS/DR1/color-z-zg-g', hips_width, hips_height, 2*self.pix_size*self.bkg_rad[1]/3600, self.sky.ra.value, self.sky.dec.value)
        url = fits.open(url)
        self.ps1 = np.zeros((hips_width, hips_height, 3))
        self.ps1[:,:,0] = url[0].data[0]
        self.ps1[:,:,1] = url[0].data[1]
        self.ps1[:,:,2] = url[0].data[2]
        self.ps1 = self.ps1/self.ps1.max(axis=(0,1))
        
        url = hips.format('CDS/P/DSS2/color', hips_width, hips_height, 2*self.pix_size*self.bkg_rad[1]/3600, self.sky.ra.value, self.sky.dec.value)
        url = fits.open(url)
        self.dss = np.zeros((hips_width, hips_height, 3))
        self.dss[:,:,0] = url[0].data[0]
        self.dss[:,:,1] = url[0].data[1]
        self.dss[:,:,2] = url[0].data[2]
        self.dss = self.dss/self.dss.max(axis=(0,1))        
        
    def mk_postcard(self, ctr):
        self.out = '{0:s}_xrt_{1:03d}'.format(self.oi, ctr+1)
        ouf_name = ouf + self.out + '.pdf'
        oud_name = oud + self.out + '_lc'
        fig = plt.figure(figsize=(16, 12))
#
        # Light curve
        ax = plt.subplot2grid((2,3), (0,0), colspan=1, rowspan=1)
        tmp = np.empty(2*self.tlc.size-2)
        tmp[:-1:2] = self.tlc[:-1]
        tmp[1::2] = self.tlc[1:]
        shift = int(tmp[0]//1000*1000)
        tmp = tmp - shift
        ax.axvline(self.tb[1]-shift, c='gray')
        ax.axvline(self.tb[2]-shift, c='gray')
        plt.plot(tmp, self.src_lc.repeat(2), label='Source', color='greenyellow')
        plt.plot(tmp, self.bkg_lc.repeat(2), label='Background', color='gold')
        plt.plot(tmp, self.sub_lc.repeat(2), label='Subtracted', color='k')
        ax.set_title(self.out)
        ax.legend()
        ax.set_xlabel('Time-{0:d} (s)'.format(shift))
        ax.set_ylabel('Rate (cts/s)')
        ax.set_ylim(bottom=0)

        # Spectrum
        ax = plt.subplot2grid((2,3), (0,1), colspan=1, rowspan=1)
        tmp = np.empty(2*spec_bin.size-2)
        tmp[:-1:2] = spec_bin[:-1]
        tmp[1::2] = spec_bin[1:]
        plt.loglog(tmp, self.src_spec.repeat(2)+0.1, label='Source')
        plt.loglog(tmp, self.bkg_spec.repeat(2)+0.1, label='Background')
        ax.legend()
        ax.set_xlabel('Energy (keV)'.format(shift))
        ax.set_ylabel('Flux density (cts/keV)')
        
        # Zoomed image
        for ii in range(0,2):
            ax = plt.subplot2grid((2,3), (ii,2), colspan=1, rowspan=1)
            plt.imshow(gaussian_filter(self.img.T, ii), cmap='afmhot', origin='lower')
            tmp = plt.Circle((self.bkg_rad[1]/binning-0.5, self.bkg_rad[1]/binning-0.5), self.src_rad/binning, color='w', lw=2, fill=False)
            ax.add_artist(tmp)
            tmp = plt.Circle((self.bkg_rad[1]/binning-0.5, self.bkg_rad[1]/binning-0.5), self.bkg_rad[0]/binning, color='w', lw=1, fill=False)
            ax.add_artist(tmp)
            tmp = plt.Circle((self.bkg_rad[1]/binning-0.5, self.bkg_rad[1]/binning-0.5), self.bkg_rad[1]/binning, color='w', lw=1, fill=False)
            ax.add_artist(tmp)
            ax.set_axis_off()
        
        # HiPS
        ax = plt.subplot2grid((2,3), (1, 0), colspan=1, rowspan=1)
        ax.imshow(self.ps1, origin='lower', interpolation='nearest')
        ax.set_axis_off()
        ax = plt.subplot2grid((2,3), (1, 1), colspan=1, rowspan=1)
        ax.imshow(self.dss, origin='lower', interpolation='nearest')
        ax.set_axis_off()
        
        # Meta
        plt.text(0, 0, self.viz + self.sim, fontdict={'family': 'monospace', 'size': 12}, transform=plt.gcf().transFigure, va='top')
        plt.text(0, 1, self.buf, fontdict={'family': 'monospace', 'size': 14}, transform=plt.gcf().transFigure, va='bottom')
        
        plt.savefig(ouf_name, bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.close()
        np.save(oud_name.replace('.pdf', '_lc'), np.c_[self.tlc[:-1], self.src_lc, self.bkg_lc])
        return 1










class Source:
    def __init__(self, obs, obj):
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
        buf = buf + 'Exposure: {0:3.0f} ks, Detection statistic: {1:6.1f}'.format((obs.tt.max()-obs.tt.min())/1.e3, self.stat) + '\n'

        self.viz = viz
        self.sim = sim
        self.buf = buf
        self.src_rad = 30/self.pix_size
        self.bkg_rad = [2*self.src_rad, 3*self.src_rad]
        self.backscal = self.src_rad**2/(self.bkg_rad[1]**2-self.bkg_rad[0]**2)

    def ext_evt(self, obs):
        def get_src_bkg_idx(self, tt, xx, yy): # For light curves and spectra
            rad = (xx-self.coo[0])**2+(yy-self.coo[1])**2
            src = (rad < self.src_rad**2)
            bkg = (rad > self.bkg_rad[0]**2) & (rad < self.bkg_rad[1]**2)
            return src, bkg

        def get_img_idx(self, tt, xx, yy): # For images
            idx = (xx > self.coo[0]-self.bkg_rad[1]) & (xx < self.coo[0]+self.bkg_rad[1]+1e-3)
            idx = (yy > self.coo[1]-self.bkg_rad[1]) & (yy < self.coo[1]+self.bkg_rad[1]+1e-3)
            return idx

        # For light curves and spectra
        src, bkg = get_src_bkg_idx(self, obs.tt, obs.xx, obs.yy)
        self.src_tt = obs.tt[src]
        self.src_pi = obs.ee[src]
        self.bkg_tt = obs.tt[bkg]
        self.bkg_pi = obs.ee[bkg]

        order = np.argsort(self.src_tt)
        self.src_tt = self.src_tt[order]
        self.src_pi = self.src_pi[order]
        order = np.argsort(self.bkg_tt)
        self.bkg_tt = self.bkg_tt[order]
        self.bkg_pi = self.bkg_pi[order]
        
        # For images
        idx = get_img_idx(self, obs.tt, obs.xx, obs.yy)
        self.xx = obs.xx[idx]
        self.yy = obs.yy[idx]
        
    def mk_img(self):
        xbin = np.arange(self.coo[0]-self.bkg_rad[1], self.coo[0]+self.bkg_rad[1]+1e-3, binning)
        ybin = np.arange(self.coo[1]-self.bkg_rad[1], self.coo[1]+self.bkg_rad[1]+1e-3, binning)
        self.img, _, _ = np.histogram2d(self.xx, self.yy, [xbin, ybin])

    def mk_lc(self):
        self.tlc = np.arange(self.src_tt[0], self.src_tt[-1], lc_bin)
        self.src_lc, _ = np.histogram(self.src_tt, self.tlc)
        self.src_lc = self.src_lc/np.diff(self.tlc)
        self.bkg_lc, _ = np.histogram(self.bkg_tt, self.tlc)
        self.bkg_lc = self.backscal*self.bkg_lc/np.diff(self.tlc)
        self.sub_lc = self.src_lc - self.bkg_lc

        
    def mk_pds(self):
        def fft_hlp(tt, dt, nn, ii):
            dc = tt.size
            if dc < 0.01*nn:
                tmp = np.zeros(int(np.ceil((tt[-1]-tt[0])/dt)))
                tmp[((tt-tt[0])//dt).astype(np.int)] += 1 # assumes only 1 photon per bin
            else:
                tmp, _ = np.histogram(tt, np.arange(tt[0], tt[-1], dt))
            
            lc = np.ones(nn)*tmp.mean()
            # discard an event or two that is accelerated out of the power-of-two array
            lc[:np.min((tmp.size, lc.size))] = tmp[:np.min((tmp.size, lc.size))]
            ft = np.fft.rfft(lc)
            po = np.abs(ft)**2/dc
            po[:ii] = 0.
            return po

        
        print('Computing PDS')
        self.nu = []
        self.po = []
        
        for jj, dt in enumerate(dT):
            nn = int(np.ceil(np.log2((self.src_tt[-1]-self.src_tt[0])/dt))+1)
            nn = min(nn, 28)
            nn = 2**nn
            self.nu.append(np.fft.rfftfreq(nn, dt))
            ii = np.argmax(self.nu[jj] > nu_min)
            self.po.append(fft_hlp(self.src_tt, dt, nn, ii))

        print('PDS completed')
    
    def mk_spec(self):
        self.src_spec, _ = np.histogram(pi2kev(self.src_pi), spec_bin)
        self.bkg_spec, _ = np.histogram(pi2kev(self.bkg_pi), spec_bin)
        self.src_spec = self.src_spec/np.diff(spec_bin)
        self.bkg_spec = self.backscal*self.bkg_spec/np.diff(spec_bin)

    def get_hips(self):
        def hips_hlp(url):
            # Recursive pauser to avoid blacklisting
            try:
                url = fits.open(url)
            except:
                print('WARNING: HIPS failed')
                url = hips_hlp(url)
            return url
        
        # https://aladin.u-strasbg.fr/hips/list
        url = hips.format('CDS/P/PanSTARRS/DR1/color-z-zg-g', hips_width, hips_height, 2*self.pix_size*self.bkg_rad[1]/3600, self.sky.ra.value, self.sky.dec.value)
        url = hips_hlp(url)
        self.ps1 = np.zeros((hips_width, hips_height, 3))
        self.ps1[:,:,0] = url[0].data[0]
        self.ps1[:,:,1] = url[0].data[1]
        self.ps1[:,:,2] = url[0].data[2]
        self.ps1 = self.ps1/self.ps1.max(axis=(0,1))
        
        url = hips.format('CDS/P/DSS2/color', hips_width, hips_height, 2*self.pix_size*self.bkg_rad[1]/3600, self.sky.ra.value, self.sky.dec.value)
        url = hips_hlp(url)
        self.dss = np.zeros((hips_width, hips_height, 3))
        self.dss[:,:,0] = url[0].data[0]
        self.dss[:,:,1] = url[0].data[1]
        self.dss[:,:,2] = url[0].data[2]
        self.dss = self.dss/self.dss.max(axis=(0,1))        
        
    def mk_postcard(self, ctr):
        self.out = '{0:s}_src_{1:03d}'.format(self.oi, ctr+1)
        ouf_name = ouf + self.out + '.pdf'
        oud_name = oud + self.out + '_pds'
        fig = plt.figure(figsize=(16, 12))
        
        # Light curve
        ax = plt.subplot2grid((2,3), (0,0), colspan=1, rowspan=1)
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
        ax = plt.subplot2grid((2,3), (0,1), colspan=1, rowspan=1)
        tmp = np.empty(2*spec_bin.size-2)
        tmp[:-1:2] = spec_bin[:-1]
        tmp[1::2] = spec_bin[1:]
        plt.loglog(tmp, self.src_spec.repeat(2)+0.1, label='Source')
        plt.loglog(tmp, self.bkg_spec.repeat(2)+0.1, label='Background')
        ax.legend()
        ax.set_xlabel('Energy (keV)'.format(shift))
        ax.set_ylabel('Flux density (cts/keV)')

        # PDS
        ax = plt.subplot2grid((2,3), (0,2), colspan=1, rowspan=1)
        ii = self.po[0] > po_min
        plt.loglog(self.nu[0][ii], self.po[0][ii],          'o', label='Both, dt = {0:6.4f} s'.format(dT[0]))
        np.save(oud_name, np.c_[self.nu[0][ii], self.po[0][ii]])
        
        ii = self.po[1] > po_min
        plt.loglog(self.nu[1][ii], self.po[1][ii],          'o', label='Both, dt = {0:6.4f} s'.format(dT[1]))
        
        ax.legend()
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Fourier power')
        ax.set_xlim([nu_min, 1/(2*min(dT))])
        ax.set_ylim(bottom=10)

        po_max = np.amax((self.po[0].max(), self.po[1].max()))
        if po_max > 22:
            print('High Fourier power: {0:.1f}!'.format(po_max))
            self.buf = self.buf[:-2] + ', High Fourier power: {0:.1f}\n'.format(po_max)
        
        # Zoomed image
        ax = plt.subplot2grid((2,3), (1,0), colspan=1, rowspan=1)
        plt.imshow(self.img.T, cmap='afmhot', origin='lower')
        tmp = plt.Circle((self.bkg_rad[1]/binning-0.5, self.bkg_rad[1]/binning-0.5), self.src_rad/binning, color='w', lw=2, fill=False)
        ax.add_artist(tmp)
        tmp = plt.Circle((self.bkg_rad[1]/binning-0.5, self.bkg_rad[1]/binning-0.5), self.bkg_rad[0]/binning, color='w', lw=1, fill=False)
        ax.add_artist(tmp)
        tmp = plt.Circle((self.bkg_rad[1]/binning-0.5, self.bkg_rad[1]/binning-0.5), self.bkg_rad[1]/binning, color='w', lw=1, fill=False)
        ax.add_artist(tmp)
        ax.set_axis_off()

        # HiPS
        ax = plt.subplot2grid((2,3), (1, 1), colspan=1, rowspan=1)
        ax.imshow(self.ps1, origin='lower', interpolation='nearest')
        ax.set_axis_off()
        ax = plt.subplot2grid((2,3), (1, 2), colspan=1, rowspan=1)
        ax.imshow(self.dss, origin='lower', interpolation='nearest')
        ax.set_axis_off()
        
        # Meta
        plt.text(0, 0, self.viz + self.sim, fontdict={'family': 'monospace', 'size': 12}, transform=plt.gcf().transFigure, va='top')
        plt.text(0, 1, self.buf, fontdict={'family': 'monospace', 'size': 14}, transform=plt.gcf().transFigure, va='bottom')
        
        plt.savefig(ouf_name, bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.close()
        np.save(oud_name.replace('pds', 'lc'), np.c_[self.tlc[:-1], self.src_lc, self.bkg_lc])
        return 1










def get_viz(coo):
    def viz_hlp(coo, ts):
        # Recursive pauser to avoid blacklisting
        try:
            viz = Vizier(columns=['*', '+_r'], timeout=60, row_limit=30).query_region(coo, radius=20*u.arcsec, catalog='I/345/gaia2,II/246/out,V/147/sdss12')
        except:
            print('Vizier blacklisted, pausing for', ts, 's.')
            time.sleep(ts)
            viz = viz_hlp(coo, 2*ts)
        return viz

    viz = viz_hlp(coo, 8)
    if len(viz) == 0:
        return ''

    # Some formating
    tmp = []
    plx_star = False
    if 'I/345/gaia2' in viz.keys():
        tmp = tmp + ['Gaia DR2'] + viz['I/345/gaia2'][gaia_col].pformat(max_width=-1)
    if 'II/246/out' in viz.keys():
        tmp = tmp + ['\n2MASS'] + viz['II/246/out'][mass_col].pformat(max_width=-1)
    if 'V/147/sdss12' in viz.keys():
        tmp = tmp + ['\nSDSS12'] + viz['V/147/sdss12'][sdss_col].pformat(max_width=-1)
    return '\n'.join(tmp)

def get_sim(coo):
    def sim_hlp(coo, ts):
        # Recursive pauser to avoid blacklisting
        try:
            sim = Simbad.query_region(coo, radius=20*u.arcsec)
        except:
            print('SIMBAD blacklisted, pausing for', ts, 's.')
            time.sleep(ts)
            sim = sim_hlp(coo, 2*ts)
        return sim

    sim = sim_hlp(coo, 8)
    if sim is None or len(sim) == 0:
        return ''
    return '\n\nSIMBAD\n' + '\n'.join(sim.pformat(max_lines=-1, max_width=-1)[:13])








    


        

def download_data(oi, ii):
    def dl_hlp(oi, ra, de):
        cc = oi[2] + '00000'
        subprocess.call(cmd + [wge.format(cc, oi)])
        subprocess.call(cmd + [orb.format(cc, oi)])
        subprocess.call(['gunzip'] + ['{0:s}_bas.fits.Z'.format(oi)])
        subprocess.call(['gunzip'] + ['{0:s}_anc.fits.Z'.format(oi)])

        proc = subprocess.Popen('rosbary', stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        tmp = '{0:s}_anc.fits\n{0:s}_tmp.fits\n{0:s}_bas.fits\n{0:s}_bc.fits\n{1:s}\n{2:s}\n'
        tmp = tmp.format(oi, str(ra), str(de))
        proc.stdin.write(tmp.encode("utf-8"))
        proc.stdin.flush()
        proc.communicate()
        proc.stdin.close()
        proc.terminate()

        subprocess.call(['rm'] + ['{0:s}_bas.fits'.format(oi)] + ['{0:s}_anc.fits'.format(oi)] + ['{0:s}_tmp.fits'.format(oi)], stderr=subprocess.DEVNULL)

        return True

    ra = coo[ii].ra.degree
    de = coo[ii].dec.degree
    
    if os.path.isfile(ffn.format(oi)):
        print('Data found on disk')
        return True
    # retcode = subprocess.call(['rm', '-rf', cwd + oi])
    # retcode = subprocess.call(['rm'] + glob('*.gz'), stderr=subprocess.DEVNULL)

    return dl_hlp(oi, ra, de)



def read_ros(ff):
    print('\nReading:', ff)
    ff = open(ff)
    log = {}
    obs = []
    tar = []
    coo = []
    tt0 = []
    exp = []
    for ll in ff:
        if ll[0] == '#':
            continue

        ll = ll.split('|')
        if not 'PSPC' in ll[2]:
            continue
        obs.append(ll[1].strip().lower())
        exp.append(float(ll[3].strip()))
        coo.append(ll[4].strip() + ' ' + ll[5].strip())
        tar.append(ll[6].strip())
        tt0.append('1990-01-01 00:00:00' if ll[7].strip() == '' else ll[7].strip())

    obs = np.array(obs)
    target = np.array(tar)
    coo = SkyCoord(coo, unit=(units.hourangle, units.degree))
    epoch = Time(np.array(tt0))
    exp = np.array(exp)
    return obs, target, coo, epoch, exp


################################################################
# Just read the data
obs, target, coo, epoch, exp = read_ros('/Users/silver/box/phd/pro/ros/sbo/dat/pspc.txt')
# cwd = '/Volumes/pow/dat/nus/cat/'
cwd = '/Users/silver/out/ros/cat/'
ouf = '/Users/silver/out/ros/fig/'
oud = '/Users/silver/out/ros/dat/'
retcode = subprocess.call(['mkdir', '-p', cwd])
retcode = subprocess.call(['mkdir', '-p', ouf])
retcode = subprocess.call(['mkdir', '-p', oud])
os.chdir(cwd)

cmd = ['curl', '--ftp-ssl', '-sS', '-O', '-J']
wge = 'ftp://legacy.gsfc.nasa.gov/rosat/data/pspc/processed_data/{0:s}/{1:s}/{1:s}_bas.fits.Z'
orb = 'ftp://legacy.gsfc.nasa.gov/rosat/data/pspc/processed_data/{0:s}/{1:s}/{1:s}_anc.fits.Z'
ffn = '{0:s}_bc.fits'
hips = 'http://alasky.u-strasbg.fr/hips-image-services/hips2fits?hips={}&width={}&height={}&fov={}&projection=TAN&coordsys=icrs&ra={}&dec={}'

img_size = 128 # pixel
binning = 20 # pixel, how much to bin the zoomed image (1 is default, results in a pixel size of 0.5 arcsec)
sbin = 100 # pixel, pixel size of 0.5 arcsec
hips_width = 256
hips_height = 256
ene_low = 0.1 # keV
ene_high = 2.0 # keV
lc_bin = 100 # s (for source light curves)
dT = [0.0005, 0.01] # s
nu_min = 1. # Hz
po_min = 20 # Fourier power
tbins = np.logspace(2,4,3) # s
test_stat_lim = 25e9 # Used 25 for NuSTAR
FLUX_LIM = 100.
spec_bin = np.logspace(np.log10(ene_low), np.log10(ene_high), 4) # keV
stat_bin = np.linspace(0, 1000, 1001)
stat_bin[0] = 1.e-39 # Padding to exclude flagged bins that are set to < 1.e-40
lc_nbin = 50 # number of bins for light curves
plim = 1-0.999999426696856 # 5 sigma

################################################################
star = int(sys.argv[1])
stop = int(sys.argv[2])

for ii, oi in enumerate(obs[star:stop]):
    # 3338/4669 (3337 doesn't work, see below, and 4043?)
    # (4650 and later lack temporal information)

    # Housekeeping
    ii = ii + star
    print('\n\n\nProcessing Obs. ID: {0:11s} (Index: {1:d})'.format(oi, ii))
    print('Downloading data')
    
    if oi[:2] == 'wp':
        print('Skipping {0:s}, German data processing'.format(oi))
        continue
    elif len(oi) < 11:
        print('Skipping {0:s}, old data processing'.format(oi))
        continue
    
    if download_data(oi, ii):
        observation = Observation(ii)
        observation.find_transients()
        observation.gen_products()
        observation.find_sources()
        del(observation)
    else:
        print('WARNING No data found for Obs. ID:', oi)




# *** urllib.error.HTTPError: HTTP Error 500: Internal Server Error
# (Pdb) url
# 'http://alasky.u-strasbg.fr/hips-image-services/hips2fits?hips=CDS/P/PanSTARRS/DR1/color-z-zg-g&width=256&height=256&fov=0.04999999999999999&projection=TAN&coordsys=icrs&ra=357.35088874947866&dec=-31.432754342848956'
