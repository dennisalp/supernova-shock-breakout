'''
2019-09-17, Dennis Alp, dalp@kth.se

Search for SBOs in the XMM archives.
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
from scipy.optimize import curve_fit
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
        print('Reading data')        
        self.ii = ii
        self.oi = obs[ii]
        self.target = target[ii]
        self.epoch = epoch[ii]
        self.tt = np.empty(0)
        self.ee = np.empty(0)
        self.xx = np.empty(0)
        self.yy = np.empty(0)
        self.tt_cam = {'PN': np.empty(0), 'M1': np.empty(0), 'M2': np.empty(0)}
        self.ee_cam = {'PN': np.empty(0), 'M1': np.empty(0), 'M2': np.empty(0)}
        self.xx_cam = {'PN': np.empty(0), 'M1': np.empty(0), 'M2': np.empty(0)}
        self.yy_cam = {'PN': np.empty(0), 'M1': np.empty(0), 'M2': np.empty(0)}
        self.keys = ['PN', 'M1', 'M2']
        
        for ff in glob(cwd + self.oi + '/*FTZ'):
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
                print('ERROR Unknown instrument:', cam, self.oi)
                sys.exit(1)
                
            good = good & (dd['PI'] > ene_low*1.e3) & (dd['PI'] < ene_high*1.e3)
            if self.oi == '0655343801':
                good = good & (dd['X'] > 190000.) & (dd['X'] < 240000.)
                good = good & (dd['Y'] > 285000.) & (dd['Y'] < 330000.)
            else:
                good = good & (dd['X'] > 0.) # XMM puts some invalid photons at (-99999999.0, -99999999.0)
                good = good & (dd['Y'] > 0.)
                good = good & (dd['X'] < 100000.+dd['X'][good].mean())
                good = good & (dd['Y'] < 100000.+dd['Y'][good].mean())
            
            self.tt_cam[self.cam] = np.concatenate((self.tt_cam[self.cam], dd['Time'][good]))
            self.ee_cam[self.cam] = np.concatenate((self.ee_cam[self.cam], dd['PI'][good]/1.e3))
            self.xx_cam[self.cam] = np.concatenate((self.xx_cam[self.cam], dd['X'][good]))
            self.yy_cam[self.cam] = np.concatenate((self.yy_cam[self.cam], dd['Y'][good]))

        for kk in self.keys:
            self.tt = np.concatenate((self.tt, self.tt_cam[kk]))
            self.ee = np.concatenate((self.ee, self.ee_cam[kk]))
            self.xx = np.concatenate((self.xx, self.xx_cam[kk]))
            self.yy = np.concatenate((self.yy, self.yy_cam[kk]))

        self.wcs = get_wcs(ff)

    def find_sources(self):
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
                cubes[:,:,:,jj], _ = np.histogramdd(np.c_[self.xx, self.yy, self.tt], bins=bins)

            return cubes

        def get_likelihood(cubes):
            # Current cell
            obs = cubes[1:-1, 1:-1, 1:-1, :]
            obs = np.where(obs < 20, 0, obs)

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
            obs = np.where(obs < 1.e-300, 1e-300, obs)
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
            np.savetxt(figp + self.oi + '_stat.txt', tmp, fmt='%i')


            
            # Find peaks
            xx, yy = np.unravel_index(np.argsort(img.ravel())[::-1], img.shape)
            obj = []
            for jj in range(0, xx.size):
                val = img[xx[jj], yy[jj]]
                if val < -1.e98:
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
                gd = gd & (self.tt < oo['t'][2]) & (self.tt > oo['t'][1])
                xb = np.linspace(oo['x'][0], oo['x'][1], nn+1)
                yb = np.linspace(oo['y'][0], oo['y'][1], nn+1)
                im, _, _ = np.histogram2d(self.xx[gd], self.yy[gd], bins=(xb, yb))
                

                ang_res = 3000/(50*sbin/nn) # Angular resolution of XMM in current pixel units
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
                oo['x'] = [np.interp(pars[1]+0.5, np.arange(xb.size), xb), err[0]*(50*sbin/nn)/1e3]
                oo['y'] = [np.interp(pars[0]+0.5, np.arange(yb.size), yb), err[1]*(50*sbin/nn)/1e3]

                ra,de=self.wcs.wcs_pix2world(oo['x'][0], oo['y'][0], 1)
                oo['sky'] = SkyCoord(ra*u.deg, de*u.deg)

            return obj



        ################################################################
        print('Finding sources')
        self.xbin = np.arange(self.xx.min(), self.xx.max(), sbin)
        self.ybin = np.arange(self.yy.min(), self.yy.max(), sbin)
        self.tbin = []
        for tbin in tbins:
            tmp = np.arange(self.tt.min(), self.tt.max(), tbin)
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
        print('Checking {0:d} source(s) and generating products'.format(len(self.obj)))
        ctr = 0
        for oi, obj in enumerate(self.obj):
            src = Source(self, obj)
            if src.keep(ctr):
                ctr += 1
                src.ext_evt(self)
                src.mk_img()
                src.mk_lc()
                src.mk_spec()
                src.mk_postcard()
        print('Generated {0:d} product(s)'.format(ctr))


        
class Source:
    def __init__(self, obs, obj):
        self.tb = obj['t']
        self.x0 = obj['x']
        self.y0 = obj['y']
        self.sky = obj['sky']
        self.keys = obs.keys
        xx, yy = np.array(obs.wcs.all_world2pix(self.sky.ra, self.sky.dec, 1))
        self.coo = np.array([xx, yy])        
        self.stat = obj['stat']
        self.oi = obs.oi

        # Query catalogs
        self.nearest = np.empty(3)
        viz, self.plx_star, self.nearest[0] = get_viz(self.sky)
        sim, self.nearest[1], self.good_type, self.agn = get_sim(self.sky)
        ned, self.nearest[2] = get_ned(self.sky)

        # Parse meta data
        self.target = target
        buf = 100*'#' + '\nIndex ' + str(obs.ii) + '\n' + 100*'#' + '\n\n'
        buf = buf + 'Obs. Date: ' + obs.epoch + '\n'
        buf = buf + 'Obs. ID: ' + obs.oi + ', Target: ' + obs.target + '\n'
        buf = buf + 'RA, Dec:' + '{0:>16s}{1:>16s}'.format(*self.sky.to_string('hmsdms').split(' '))+ '\n'
        buf = buf + 'RA, Dec:' + '{0:>16s}{1:>16s}'.format(*self.sky.to_string('decimal').split(' ')) + '\n'
        buf = buf + 'l, b:   ' + '{0:>16s}{1:>16s}'.format(*self.sky.galactic.to_string('decimal').split(' ')) + '\n'
        astrometry = np.sqrt(self.x0[1]**2+self.y0[1]**2)
        if astrometry < 30:
            buf = buf + 'Astrometric accuracy: {0:7.3f} arcsec\n'.format(astrometry)
        else:
            buf = buf + 'Astrometric accuracy undetermined; fitting failed\n'
        buf = buf + 'Exposure: {0:3.0f} ks, Detection statistic: {1:6.1f}'.format((obs.tt.max()-obs.tt.min())/1.e3, self.stat) + '\n'

        self.viz = viz
        self.sim = sim
        self.ned = ned
        self.buf = buf
        self.src_rad = 400 # pixels, pixel size of 50 mas
        self.bkg_rad = [800, 1200]
        self.backscal = self.src_rad**2/(self.bkg_rad[1]**2-self.bkg_rad[0]**2)

    def keep(self, ctr):
        if self.good_type and not self.plx_star:
            self.out = '{0:s}_{1:03d}'.format(self.oi, ctr+1)
            return True
        return False

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
        
        self.xx_cam = {}
        self.yy_cam = {}
        for kk in self.keys:
            idx = get_img_idx(self, obs.tt_cam[kk], obs.xx_cam[kk], obs.yy_cam[kk])
            self.xx_cam[kk] = obs.xx_cam[kk][idx]
            self.yy_cam[kk] = obs.yy_cam[kk][idx]

    def mk_img(self):
        xbin = np.arange(self.coo[0]-self.bkg_rad[1], self.coo[0]+self.bkg_rad[1]+1e-3, binning)
        ybin = np.arange(self.coo[1]-self.bkg_rad[1], self.coo[1]+self.bkg_rad[1]+1e-3, binning)
        self.img, _, _ = np.histogram2d(self.xx, self.yy, [xbin, ybin])
        self.img_cam = {}
        for kk in self.keys:
            self.img_cam[kk], _, _ = np.histogram2d(self.xx_cam[kk], self.yy_cam[kk], [xbin, ybin])

    def mk_lc(self):
        self.tlc = np.linspace(self.tb[0], self.tb[3], lc_nbin+1)
        self.src_lc, _ = np.histogram(self.src_tt, self.tlc)
        self.src_lc = self.src_lc/np.diff(self.tlc)
        self.bkg_lc, _ = np.histogram(self.bkg_tt, self.tlc)
        self.bkg_lc = self.backscal*self.bkg_lc/np.diff(self.tlc)
        self.sub_lc = self.src_lc - self.bkg_lc
        
    def mk_spec(self):
        self.src_spec, _ = np.histogram(self.src_pi, spec_bin)
        self.bkg_spec, _ = np.histogram(self.bkg_pi, spec_bin)
        self.src_spec = self.src_spec/np.diff(spec_bin)
        self.bkg_spec = self.backscal*self.bkg_spec/np.diff(spec_bin)

    def mk_postcard(self):
        out_name = figp + self.out + '.pdf'
        fig = plt.figure(figsize=(16, 12))
#
        # Light curve
        ax = plt.subplot2grid((3,3), (0,0), colspan=1, rowspan=1)
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
        ax = plt.subplot2grid((3,3), (0,1), colspan=1, rowspan=1)
        tmp = np.empty(2*spec_bin.size-2)
        tmp[:-1:2] = spec_bin[:-1]
        tmp[1::2] = spec_bin[1:]
        plt.loglog(tmp, self.src_spec.repeat(2)+0.1, label='Source')
        plt.loglog(tmp, self.bkg_spec.repeat(2)+0.1, label='Background')
        ax.legend()
        ax.set_xlabel('Energy (keV)'.format(shift))
        ax.set_ylabel('Flux density (cts/keV)')
        
        # Zoomed image
        ax = plt.subplot2grid((3,3), (0,2), colspan=1, rowspan=1)
        plt.imshow(self.img.T, cmap='afmhot', origin='lower')
        tmp = plt.Circle((self.bkg_rad[1]/binning-0.5, self.bkg_rad[1]/binning-0.5), self.src_rad/binning, color='w', lw=2, fill=False)
        ax.add_artist(tmp)
        tmp = plt.Circle((self.bkg_rad[1]/binning-0.5, self.bkg_rad[1]/binning-0.5), self.bkg_rad[0]/binning, color='w', lw=1, fill=False)
        ax.add_artist(tmp)
        tmp = plt.Circle((self.bkg_rad[1]/binning-0.5, self.bkg_rad[1]/binning-0.5), self.bkg_rad[1]/binning, color='w', lw=1, fill=False)
        ax.add_artist(tmp)
        ax.set_axis_off()

        for ii in range(2*len(self.keys)):
            kk = self.keys[np.mod(ii,len(self.img_cam.keys()))]
            ax = plt.subplot2grid((3,3), (ii//3+1, np.mod(ii,3)), colspan=1, rowspan=1)
            tmp = self.img_cam[kk].T
            tmp = gaussian_filter(tmp, 2*(ii//3))
            plt.imshow(tmp, cmap='afmhot', origin='lower')
            tmp = plt.Circle((self.bkg_rad[1]/binning-0.5, self.bkg_rad[1]/binning-0.5), self.src_rad/binning, color='w', lw=2, fill=False)
            ax.add_artist(tmp)
            tmp = plt.Circle((self.bkg_rad[1]/binning-0.5, self.bkg_rad[1]/binning-0.5), self.bkg_rad[0]/binning, color='w', lw=1, fill=False)
            ax.add_artist(tmp)
            tmp = plt.Circle((self.bkg_rad[1]/binning-0.5, self.bkg_rad[1]/binning-0.5), self.bkg_rad[1]/binning, color='w', lw=1, fill=False)
            ax.add_artist(tmp)
            ax.set_axis_off()

        
        # Meta
        plt.text(0, 0, self.viz + self.sim + self.ned, fontdict={'family': 'monospace', 'size': 12}, transform=plt.gcf().transFigure, va='top')
        plt.text(0, 1, self.buf, fontdict={'family': 'monospace', 'size': 14}, transform=plt.gcf().transFigure, va='bottom')
        
        plt.savefig(out_name, bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.close()
        np.savetxt(out_name.replace('.pdf', '.txt'), np.c_[self.tlc[:-1], self.src_lc, self.bkg_lc])
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
    
def download_data(oi):
    def dl_hlp(cmd):
        retcode = subprocess.call(cmd)
        gtar = glob('GUEST*.tar')
        gftz = glob('*.FTZ')
        file_missing = ' '.join(cmd).split('/')[-1]
        if len(gtar) == 1:
            retcode = subprocess.call(['tar', '-xvf', cwd + gtar[0], '-C', cwd], stderr=subprocess.DEVNULL)
            retcode = subprocess.call(['mv'] + glob(cwd + oi + '/pps/*') + [cwd + oi])
            retcode = subprocess.call(['rm', '-rf', cwd + oi + '/pps'])
            retcode = subprocess.call(['rm', cwd + gtar[0]])
            return True
        elif len(gftz) == 1:
            retcode = subprocess.call(['mkdir', '-p', cwd + oi])
            retcode = subprocess.call(['mv', cwd + gftz[0], cwd + oi])
            return True
        elif os.path.isfile(file_missing):
            retcode = subprocess.call(['rm', file_missing])
            return False

        print('ERROR Unknown data format delivered from XSA AIO:', oi)
        print(' '.join(cmd))
        sys.exit(1)

    if os.path.isdir(cwd + oi):
        return True
    retcode = subprocess.call(['rm', '-rf', cwd + oi])
    retcode = subprocess.call(['rm'] + glob('GUEST*.tar'), stderr=subprocess.DEVNULL)
    retcode = subprocess.call(['rm'] + glob('*.FTZ'), stderr=subprocess.DEVNULL)

    tmp = dl_hlp(cmd + [purl.format(oi)])
    return dl_hlp(cmd + [murl.format(oi)]) or tmp

def read_old_log(ff):
    ff = open(ff)
    obs = set()
    for ll in ff:
        if ll[0] == '#':
            continue
        obs.add(ll.split('|')[1].strip())
    return obs

def read_log(ff, processed):
    ff = open(ff)
    obs = []
    target = []
    epoch = []
    for ll in ff:
        if ll[0] == '#':
            continue
        if not len(ll.split('|')) == 13:
            print('Pipe (|) used for purpose other than as a delimiter (e.g. in the target name)')
            sys.exit(1)

        tmp = ll.split('|')[1].strip()
        if tmp in processed:
            processed.remove(tmp)
            continue
        obs.append(tmp)
        target.append(ll.split('|')[3].strip())
        epoch.append(ll.split('|')[6].strip())

    return obs, target, epoch



################################################################
# Just read the data
figp = '/Users/silver/out/sne/sbo/cluster/' + sys.argv[1] + '/fig/'
processed = read_old_log('/Users/silver/box/phd/pro/sne/sbo/cluster/log.txt')
obs, target, epoch = read_log('/Users/silver/Dropbox/cluster/v2/all.txt', processed)
cwd = '/Users/silver/out/sne/sbo/cluster/' + sys.argv[1] + '/dat/'
retcode = subprocess.call(['mkdir', '-p', cwd])
retcode = subprocess.call(['mkdir', '-p', figp])
os.chdir(cwd)
cmd = ['curl', '-sS', '-O', '-J']
purl = 'http://nxsa.esac.esa.int/nxsa-sl/servlet/data-action-aio?level=PPS&extension=FTZ&name=PIEVLI&obsno={0:10s}'
murl = 'http://nxsa.esac.esa.int/nxsa-sl/servlet/data-action-aio?level=PPS&extension=FTZ&name=MIEVLI&obsno={0:10s}'
img_size = 1024 # pixel
binning = 80 # pixel, how much to bin the zoomed image (80 is default, results in a pixel size of 4 arcsec), i.e. pixel scale of 50 mas
sbin = 400 # spatial bin for source finding, pixel size of 20 arcsec
ene_low = 0.3 # keV
ene_high = 2.0 # keV
tbins =  [300, 1000, 3000] # s
tbins = np.logspace(2,4,5) # s
test_stat_lim = 25
spec_bin = np.logspace(np.log10(ene_low), np.log10(ene_high) , 8) # keV
stat_bin = np.linspace(0, 1000, 1001)
stat_bin[0] = 1.e-39 # Padding to exclude flagged bins that are set to < 1.e-40
lc_nbin = 50 # number of bins for light curves
plim = 1-0.999999426696856 # 5 sigma
XMMEA_EP = 65584
XMMEA_EM = 65000

################################################################
start = int(sys.argv[2])
stop = int(sys.argv[3])
for ii, oi in enumerate(obs[start:stop]):
    # Housekeeping
    ii = ii + start
    print('\nProcessing Obs. ID: {0:10s} (Index: {1:d})'.format(oi, ii))
    print('Downloading data')

    if download_data(oi):
        observation = Observation(ii)
        observation.find_sources()
        observation.gen_products()
        retcode = subprocess.call(['rm', '-rf', cwd + obs[ii]])
        del(observation)
    else:
        print('WARNING No data found for Obs. ID:', oi, '(could be an RGS/OM only observation)')
    
#db()
