from pdb import set_trace as st
from datetime import date
from glob import glob
import os
import subprocess
import sys
import time

from astropy import units as u
from astropy.coordinates import SkyCoord
from scipy.ndimage import gaussian_filter
from sklearn.cluster import DBSCAN
from tqdm import tqdm
import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sts

# from astropy import units
# from astropy.time import Time
from astropy.io import fits

from catalogs import get_viz, get_sim, get_ned
from utils import get_wcs, scatter3d_obj_hlp, scatter3d_obj_plt, plt_obs_img

plt.switch_backend('agg')



ene_low = 0.3 # keV
ene_high = 10.0 # keV
sbin = 400 # spatial bin for source finding, pixel size of 20 arcsec
scale = 400  # convert unit from 50 mas to 20"
img_size = 1024  # pixel
rc_max = 1  # maximum cluster sample distance
nc_min_cam = 3  # mininum number of events in a cluster in each camera
nc_min = 6  # mininum number of pn events in a cluster
dt_just_bfaftr = 1000  # s, used to define a time period just before and after the transient
dt_pn = 0.073  # pn frame time, standard
dt_mos = 2.6  # MOS frame time, standard
binning = 80  # pixel, how much to bin the zoomed image (80 is default, results in a pixel size of 4 arcsec), i.e. pixel scale of 50 mas
spec_bin = np.logspace(np.log10(ene_low), np.log10(ene_high) , 8)  # keV
lc_nbin = 50  # number of bins for light curves
tpad = 10  # padding on either side of t0 and t1, how much to see before and after cluster



class Observation:
    def __init__(self, ii, obs, target, epoch, figp):
        print('Reading data')
        self.ii = ii
        self.oi = obs
        self.target = target
        self.epoch = epoch
        self.tt = np.empty(0)
        self.ee = np.empty(0)
        self.xx = np.empty(0)
        self.yy = np.empty(0)
        self.tt_cam = {'M1': np.empty(0), 'M2': np.empty(0), 'PN': np.empty(0)}
        self.ee_cam = {'M1': np.empty(0), 'M2': np.empty(0), 'PN': np.empty(0)}
        self.xx_cam = {'M1': np.empty(0), 'M2': np.empty(0), 'PN': np.empty(0)}
        self.yy_cam = {'M1': np.empty(0), 'M2': np.empty(0), 'PN': np.empty(0)}
        self.keys = ['M1', 'M2', 'PN']

        for ff in sorted(glob(f'{obs}/*FTZ')):
            self.cam = ff.split('/')[-1][11:13]
            dd = fits.open(ff)[1].data
            if dd.size == 0:
                print(f"WARNING No events in {ff.split('/')[-1]}, skipping")
                continue

            assert self.cam in ('M1', 'M2', 'PN'), 'ERROR Unknown instrument: '+ self.cam + ', '+ self.oi

            good = (dd['PI'] > ene_low*1.e3) & (dd['PI'] < ene_high*1.e3)
            if self.oi == '0655343801':  # The attitudes of these ones are just all messed up
                good = good & (dd['X'] > 190000.) & (dd['X'] < 240000.)
                good = good & (dd['Y'] > 285000.) & (dd['Y'] < 330000.)
            elif self.oi == '0677690101':
                good = good & (dd['X'] > -1e6) & (dd['X'] < 1e6)
                good = good & (dd['Y'] > -1e6) & (dd['Y'] < 1e6)
            else:
                good = good & (dd['X'] > 0.)  # XMM puts some invalid photons at (-99999999.0, -99999999.0)
                good = good & (dd['Y'] > 0.)
                good = good & (dd['X'] < 100000.+dd['X'][good].mean())
                good = good & (dd['Y'] < 100000.+dd['Y'][good].mean())

            tcol = 'TIME_RAW' if self.cam == 'PN' else 'TIME'
            self.tt_cam[self.cam] = np.concatenate((self.tt_cam[self.cam], dd[tcol][good]))
            self.ee_cam[self.cam] = np.concatenate((self.ee_cam[self.cam], dd['PI'][good]/1.e3))
            self.xx_cam[self.cam] = np.concatenate((self.xx_cam[self.cam], dd['X'][good]))
            self.yy_cam[self.cam] = np.concatenate((self.yy_cam[self.cam], dd['Y'][good]))

        for kk in self.keys:
            self.tt = np.concatenate((self.tt, self.tt_cam[kk]))
            self.ee = np.concatenate((self.ee, self.ee_cam[kk]))
            self.xx = np.concatenate((self.xx, self.xx_cam[kk]))
            self.yy = np.concatenate((self.yy, self.yy_cam[kk]))

        self.wcs = get_wcs(ff)

        # self.fake_source()

        figp = figp + self.oi + '.pdf'
        plt_obs_img(figp, self.xx, self.yy)



    def fake_source(self):
        if np.random.rand() < 1.:
            x0 = np.percentile(self.xx, 40)
            y0 = np.percentile(self.yy, 40)
            t0 = np.percentile(self.tt, 40)

            nn = 8
            xx = 2. * np.random.rand(3*nn) * np.cos(2*np.pi*np.random.rand(3*nn)) * scale/3 + x0
            yy = 2. * np.random.rand(3*nn) * np.sin(2*np.pi*np.random.rand(3*nn)) * scale/3 + y0
            tt = .1 * np.random.rand(3*nn) + t0*np.ones(3*nn)
            ee =    np.ones(3*nn)
            self.xx = np.concatenate((self.xx, xx))
            self.yy = np.concatenate((self.yy, yy))
            self.tt = np.concatenate((self.tt, tt))
            self.ee = np.concatenate((self.ee, ee))

            for ii, kk in enumerate(self.keys):
                self.xx_cam[kk] = np.concatenate((self.xx_cam[kk], xx[ii*nn:(ii+1)*nn]))
                self.yy_cam[kk] = np.concatenate((self.yy_cam[kk], yy[ii*nn:(ii+1)*nn]))
                self.tt_cam[kk] = np.concatenate((self.tt_cam[kk], tt[ii*nn:(ii+1)*nn]))
                self.ee_cam[kk] = np.concatenate((self.ee_cam[kk], ee[ii*nn:(ii+1)*nn]))


    def find_sources(self):
        def find_src_helper(obs, cc, dc, scale):
            r0 = cc.mean(0)
            rr = (dc[0]/2)**2 + (dc[1]/2)**2
            t0 = cc[:,2].min()
            t1 = cc[:,2].max()
            cts = np.empty(3)
            for ii, kk in enumerate(obs.keys):
                xx = self.xx_cam[kk] / scale
                yy = self.yy_cam[kk] / scale
                tt = self.tt_cam[kk]
                inside = (xx - r0[0])**2 + (yy - r0[1])**2 < rr
                dt = 1e-3 if kk == 'PN' else dt_mos/2
                during = (tt >= t0 - dt) & (tt <= t1 + dt)
                cts[ii] = (inside & during).sum()
            return cts, r0, rr

        print('Finding sources')
        # Search must be performed in pn due to its temporal resolution
        xx = self.xx_cam['PN'] / scale
        yy = self.yy_cam['PN'] / scale
        tt = self.tt_cam['PN']
        t0 = tt.min()
        t1 = tt.max()
        dt = t1 - t0
        xyt = np.column_stack((xx, yy, tt))

        # Create and fit the DBSCAN model
        print('Clustering sources')
        dbscan = DBSCAN(eps=rc_max, min_samples=nc_min, n_jobs=-1)
        cluster_labels = dbscan.fit_predict(xyt)

        print('Filtering sources')
        self.obj = []
        for ii in tqdm(range(cluster_labels.max() + 1)):
            ic = cluster_labels == ii
            nc = ic.sum()

            if nc < nc_min:  # 1. Minimum number of events in pn
                continue

            cc = xyt[ic]

            dc = cc.max(0) - cc.min(0)
            if dc[2] > 300:  # 2. Duration shorter than 300 s
                continue

            cts, r0, rr = find_src_helper(self, cc, dc, scale)
            if not np.all(cts > nc_min_cam):  # 3. Minimum number of counts in all cameras
                continue

            fr = 2
            src = (self.xx/scale - r0[0])**2 + (self.yy/scale - r0[1])**2 < rr
            during = (self.tt >= cc[:,2].min() - 1e-3) & (self.tt <= cc[:,2].max() + 1e-3)
            rate_src_bfaftr = (src & ~during).sum() / (self.tt.max() - self.tt.min() - dc[2])
            persistent = rate_src_bfaftr > 1.
            if persistent:  # 4. Persistent rate below 1 count s-1 in all cameras
                continue

            jst_bfaftr = (self.tt >= cc[:,2].min() - dt_just_bfaftr) & (self.tt <= cc[:,2].max() + dt_just_bfaftr)
            jst_bfaftr = src & jst_bfaftr & ~during
            rate_jst_bfaftr = jst_bfaftr.sum() / (2*dt_just_bfaftr)
            flaring = rate_jst_bfaftr > 2.
            if flaring:  # 5. Transient not on top of a flare of more than
                continue

            larger = (self.xx/scale - r0[0])**2 + (self.yy/scale - r0[1])**2 < rr * fr**2
            bkg = larger & ~src
            backscal = 1/(fr**2-1)

            rate_src_during = (src &  during).sum() / (dc[2] + dt_pn)
            rate_bkg_during = (bkg &  during).sum() / (dc[2] + dt_pn) * backscal
            low_sbr = rate_src_during < 3 * rate_bkg_during
            if low_sbr:  # 6. Source rate at least 3 times greater than background during the time interval
                continue

            rate_bkg_bfaftr = (bkg & ~during).sum() / (self.tt.max() - self.tt.min() - dc[2]) * backscal
            rate_net_during = rate_src_during - rate_bkg_during
            rate_net_bfaftr = rate_src_bfaftr - rate_bkg_bfaftr

            high_src = rate_src_during >  3 * rate_src_bfaftr  # 7. Source rate at least 3 times greater during the time interval than before and after
            high_net = rate_net_during > 10 * rate_net_bfaftr  # 8. Net rate at least 10 times greater during the time interval than before and after
            if high_src and high_net:
                obj = np.array([cc.min(0), cc.max(0)])
                obj[:,:2] = obj[:,:2] * scale

                x = np.concatenate((self.xx_cam['M1'], self.xx_cam['M2'], self.xx_cam['PN'][~ic]))
                y = np.concatenate((self.yy_cam['M1'], self.yy_cam['M2'], self.yy_cam['PN'][~ic]))
                t = np.concatenate((self.tt_cam['M1'], self.tt_cam['M2'], self.tt_cam['PN'][~ic]))
                mmp = np.column_stack((x/scale, y/scale, t))
                scatter3d = scatter3d_obj_hlp(xyt, mmp, cc, ic)
                self.obj.append((obj, scatter3d))


    def gen_products(self, figp):
        print('Checking {0:d} source(s) and generating products'.format(len(self.obj)))
        ctr = 0
        for oi, (obj, scatter3d) in enumerate(self.obj):
            src = Source(self, obj, scatter3d)
            if src.keep(ctr):
                ctr += 1
                src.ext_evt(self)
                src.plt_scatter(figp)
                src.mk_img()
                src.mk_lc()
                src.mk_spec()
                src.mk_postcard(figp)

        print('Generated {0:d} product(s)'.format(ctr))



class Source:
    def __init__(self, obs, obj, scatter3d):
        self.keys = obs.keys
        xx = np.mean(obj[:,0])
        yy = np.mean(obj[:,1])
        self.coo = np.array([xx, yy])
        self.x0 = obj[0,0]
        self.y0 = obj[0,1]
        self.t0 = obj[0,2]
        self.x1 = obj[1,0]
        self.y1 = obj[1,1]
        self.t1 = obj[1,2]
        ra, de = obs.wcs.all_pix2world(xx, yy, 1)
        self.sky = SkyCoord(ra*u.deg, de*u.deg)
        self.oi = obs.oi

        # Query catalogs
        self.nearest = np.empty(3)
        viz, self.plx_star, self.nearest[0] = get_viz(self.sky)
        sim, self.nearest[1], self.good_type, self.agn = get_sim(self.sky)
        ned, self.nearest[2] = get_ned(self.sky)

        # Parse meta data
        buf = 100*'#' + '\nIndex ' + str(obs.ii) + '\n' + 100*'#' + '\n\n'
        buf = buf + 'Obs. Date: ' + obs.epoch + '\n'
        buf = buf + 'Obs. ID: ' + obs.oi + ', Target: ' + obs.target + '\n'
        buf = buf + 'RA, Dec:' + '{0:>16s}{1:>16s}'.format(*self.sky.to_string('hmsdms').split(' '))+ '\n'
        buf = buf + 'RA, Dec:' + '{0:>16s}{1:>16s}'.format(*self.sky.to_string('decimal').split(' ')) + '\n'
        buf = buf + 'l, b:   ' + '{0:>16s}{1:>16s}'.format(*self.sky.galactic.to_string('decimal').split(' ')) + '\n'

        dc = (obj.max(0) - obj.min(0))
        astrometry = np.sqrt(dc[0]**2+dc[1]**2) * 50e-3
        buf = buf + 'Astrometric accuracy: {0:7.3f} arcsec\n'.format(astrometry)
        buf = buf + 'Exposure: {0:3.0f} ks, Duration: {1:6.3f} s'.format((obs.tt.max()-obs.tt.min())/1.e3, dc[2]) + '\n'

        self.viz = viz
        self.sim = sim
        self.ned = ned
        self.buf = buf
        self.src_rad = 400  # pixels, pixel size of 50 mas
        self.bkg_rad = [800, 1200]
        self.backscal = self.src_rad**2/(self.bkg_rad[1]**2-self.bkg_rad[0]**2)

        self.scatter3d = scatter3d

        during = (obs.tt > obj[0,2] - dt_mos/2) & (obs.tt < obj[1,2] + dt_mos/2)
        self.xd = obs.xx[during]
        self.yd = obs.yy[during]


    def keep(self, ctr):
        if self.good_type and not self.plx_star:
            self.out = '{0:s}_{1:03d}_god'.format(self.oi, ctr+1)
            return True
        self.out = '{0:s}_{1:03d}_bad'.format(self.oi, ctr+1)
        return True


    def plt_scatter(self, figp):
        out_name = figp + self.out + '_3d.pdf'
        scatter3d_obj_plt(self.scatter3d, out_name)
        out_name = figp + self.out + '_obs.pdf'
        plt_obs_img(out_name, self.xd, self.yd)


    def ext_evt(self, obs):
        def get_src_bkg_idx(self, tt, xx, yy):  # For spectra
            rad = (xx-self.coo[0])**2+(yy-self.coo[1])**2
            gd =  (tt >= self.t0-tpad) & (tt <= self.t1+tpad)
            src = (rad < self.src_rad**2) & gd
            bkg = (rad > self.bkg_rad[0]**2) & (rad < self.bkg_rad[1]**2) & gd
            return src, bkg

        def get_lc_idx(self, tt, xx, yy):  # For light curves
            rad = (xx-self.coo[0])**2+(yy-self.coo[1])**2
            gd =  (tt >= self.t0-tpad) & (tt <= self.t1+tpad)
            src = (rad < self.src_rad**2) & gd
            bkg = (rad > self.bkg_rad[0]**2) & (rad < self.bkg_rad[1]**2)
            return src, bkg

        def get_img_idx(self, tt, xx, yy):  # For images
            idx = (tt >= self.t0-tpad) & (tt <= self.t1+tpad)
            idx = idx & (xx > self.coo[0]-self.bkg_rad[1]) & (xx < self.coo[0]+self.bkg_rad[1]+1e-3)
            idx = idx & (yy > self.coo[1]-self.bkg_rad[1]) & (yy < self.coo[1]+self.bkg_rad[1]+1e-3)
            return idx

        # For and spectra
        src, bkg = get_src_bkg_idx(self, obs.tt, obs.xx, obs.yy)
        self.src_pi = obs.ee[src]
        self.bkg_pi = obs.ee[bkg]

        # For light curves
        src, bkg = get_lc_idx(self, obs.tt, obs.xx, obs.yy)
        self.src_tt = obs.tt[src]
        self.bkg_tt = obs.tt[bkg]

        order = np.argsort(self.src_tt)
        self.src_tt = self.src_tt[order]
        order = np.argsort(self.bkg_tt)
        self.bkg_tt = self.bkg_tt[order]

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
        self.tlc = np.linspace(-tpad, self.t1-self.t0+tpad, lc_nbin+1)
        self.src_lc, _ = np.histogram(self.src_tt-self.t0, self.tlc)
        self.src_lc = self.src_lc/np.diff(self.tlc)
        self.bkg_lc, _ = np.histogram(self.bkg_tt-self.t0, self.tlc)
        self.bkg_lc = self.backscal*self.bkg_lc/np.diff(self.tlc)
        self.sub_lc = self.src_lc - self.bkg_lc

    def mk_spec(self):
        self.src_spec, _ = np.histogram(self.src_pi, spec_bin)
        self.bkg_spec, _ = np.histogram(self.bkg_pi, spec_bin)
        self.src_spec = self.src_spec/np.diff(spec_bin)
        self.bkg_spec = self.backscal*self.bkg_spec/np.diff(spec_bin)

    def mk_postcard(self, figp):
        out_name = figp + self.out + '_pc.pdf'
        fig = plt.figure(figsize=(16, 12))

        # Light curve
        ax = plt.subplot2grid((3,3), (0,0), colspan=1, rowspan=1)
        tmp = np.empty(2*self.tlc.size-2)
        tmp[:-1:2] = self.tlc[:-1]
        tmp[1::2] = self.tlc[1:]
        ax.axvline(0., c='gray')
        ax.axvline(self.t1-self.t0, c='gray')
        plt.plot(tmp, self.src_lc.repeat(2), label='Source', color='greenyellow')
        plt.plot(tmp, self.bkg_lc.repeat(2), label='Background', color='gold')
        plt.plot(tmp, self.sub_lc.repeat(2), label='Subtracted', color='k')
        ax.set_title(self.out)
        ax.legend()
        ax.set_xlabel('Time (s)')
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
        ax.set_xlabel('Energy (keV)')
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
        np.savetxt(out_name.replace('_pc.pdf', '_lc.txt'), np.c_[self.tlc[:-1], self.src_lc, self.bkg_lc])
