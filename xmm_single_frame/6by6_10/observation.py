from pdb import set_trace as st
from datetime import date
from glob import glob
import os
import subprocess
import sys
import time

from astropy import units as u
from astropy.coordinates import SkyCoord
from scipy.ndimage import gaussian_filter, convolve
from scipy.spatial.distance import cdist
import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sts
from tqdm import tqdm

from astropy.io import fits

from catalogs import get_viz, get_sim, get_ned
from utils import get_wcs, plt_obs_img

plt.switch_backend('agg')


sig = 400  # 50 mas, Gaussian kernel sigma
src_rad = 800  # 50 mas
ene_low = 0.3  # keV
ene_high = 10.0  # keV
plim = np.log(2)+sts.norm.logsf(5)  # 5 sigma threshold for the poisson statistic
plim_mos = np.log(2)+sts.norm.logsf(3)  # 3 sigma threshold for the poisson statistic in either MOS
clim = 8  # minimum number of counts
clim_conv = 3  # minimum convolution value
rlim = 3  # minimum relative magnitude
dt_mos = 2.6  # MOS frame time, standard
dt_just_bfaftr = 100  # s, used to define a time period just before and after the transient
nn_mos_min = 3  # mininum number of events in either MOS
binning = 80  # pixel, how much to bin the zoomed image (80 is default, results in a pixel size of 4 arcsec), i.e. pixel scale of 50 mas
spec_bin = np.logspace(np.log10(ene_low), np.log10(ene_high) , 8)  # keV
lc_nbin = 50  # number of bins for light curves
tpad = 10  # padding on either side of t0 and t1, how much to see before and after cluster
XMMEA_EP = 65584
XMMEA_EM = 65000
DEBUG = 'dbg' in sys.argv

if DEBUG:
    plt.switch_backend('tkagg')



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
        self.ccd = np.empty(0)
        self.dx = np.empty(0)
        self.dy = np.empty(0)
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

            # Concerning flags
            # https://xmm-tools.cosmos.esa.int/external/sas/current/doc/eimageget/node4.html
            # http://xmm.esac.esa.int/xmmhelp/EPICpn?id=8560;expression=xmmea;user=guest
            if self.cam == 'PN':
                good = dd['FLAG'] <= XMMEA_EP
                good = good & (dd['PATTERN'] <= 4)
            elif self.cam == 'M1' or self.cam == 'M2':
                good = dd['FLAG'] <= XMMEA_EM
                good = good & (dd['PATTERN'] <= 12)

            good = good & (dd['PI'] > ene_low*1.e3) & (dd['PI'] < ene_high*1.e3)
            if self.oi == '0655343801':  # The attitudes of these ones are just all messed up
                good = good & (dd['X'] > 190000.) & (dd['X'] < 240000.)
                good = good & (dd['Y'] > 285000.) & (dd['Y'] < 330000.)
            elif self.oi == '0677690101':
                good = good & (dd['X'] > -1e6) & (dd['X'] < 1e6)
                good = good & (dd['Y'] > -1e6) & (dd['Y'] < 1e6)
            else:
                good = good & (dd['X'] > 0.) & (dd['Y'] > 0.)  # XMM puts some invalid photons at (-99999999.0, -99999999.0)
                good = good & (dd['X'] < 100000.+dd['X'][good].mean())
                good = good & (dd['Y'] < 100000.+dd['Y'][good].mean())

            tcol = 'TIME_RAW' if self.cam == 'PN' else 'TIME'
            self.tt_cam[self.cam] = np.concatenate((self.tt_cam[self.cam], dd[tcol][good]))
            self.ee_cam[self.cam] = np.concatenate((self.ee_cam[self.cam], dd['PI'][good]/1.e3))
            self.xx_cam[self.cam] = np.concatenate((self.xx_cam[self.cam], dd['X'][good]))
            self.yy_cam[self.cam] = np.concatenate((self.yy_cam[self.cam], dd['Y'][good]))

            if self.cam == 'PN':
                self.ccd = np.concatenate((self.ccd, dd.CCDNR[good]))
                self.dx  = np.concatenate((self.dx, dd.DETX[good]))
                self.dy  = np.concatenate((self.dy, dd.DETY[good]))

        for kk in self.keys:
            self.tt = np.concatenate((self.tt, self.tt_cam[kk]))
            self.ee = np.concatenate((self.ee, self.ee_cam[kk]))
            self.xx = np.concatenate((self.xx, self.xx_cam[kk]))
            self.yy = np.concatenate((self.yy, self.yy_cam[kk]))

        self.wcs = get_wcs(ff)
        self.obj = []

        # self.fake_source()

        figp = figp + self.oi + '.pdf'
        self.img, self.xb, self.yb = plt_obs_img(figp, self.xx, self.yy)


    # def fake_source(self):
    #     if np.random.rand() < 1.:
    #         x0 = np.percentile(self.xx, 43)
    #         y0 = np.percentile(self.yy, 43)
    #         t0 = np.percentile(self.tt, 43)

    #         nn = 9
    #         rr = np.random.rand(3*nn)**.5
    #         th = np.random.rand(3*nn)
    #         xx = rr * np.cos(2*np.pi*th) * sig + x0
    #         yy = rr * np.sin(2*np.pi*th) * sig + y0
    #         tt = t0*np.ones(3*nn)
    #         ee =    np.ones(3*nn)
    #         self.xx = np.concatenate((self.xx, xx))
    #         self.yy = np.concatenate((self.yy, yy))
    #         self.tt = np.concatenate((self.tt, tt))
    #         self.ee = np.concatenate((self.ee, ee))

    #         for ii, kk in enumerate(self.keys):
    #             self.xx_cam[kk] = np.concatenate((self.xx_cam[kk], xx[ii*nn:(ii+1)*nn]))
    #             self.yy_cam[kk] = np.concatenate((self.yy_cam[kk], yy[ii*nn:(ii+1)*nn]))
    #             self.tt_cam[kk] = np.concatenate((self.tt_cam[kk], tt[ii*nn:(ii+1)*nn]))
    #             self.ee_cam[kk] = np.concatenate((self.ee_cam[kk], ee[ii*nn:(ii+1)*nn]))


    def find_sources(self):
        def gaussian_2d(r1, r2, sig):
            return np.exp(-cdist(r1, r2) ** 2 / (2 * sig ** 2))

        def check_mos(self, r0, t0, t1):
            for ii, cc in enumerate(['M1', 'M2']):
                xx = self.xx_cam[cc]
                yy = self.yy_cam[cc]
                tt = self.tt_cam[cc]
                src = (xx-r0[0])**2 + (yy-r0[1])**2 < src_rad**2

                if src.sum() == 0:
                    continue

                xx = xx[src]
                yy = yy[src]
                tt = tt[src]
                during = (tt >= t0-dt_mos/2) & (tt <= t1+dt_mos/2)
                nn = during.sum()
                if nn < nn_mos_min:
                    continue

                bfaftr = (tt >= t0-dt_just_bfaftr) & (tt <= t1+dt_just_bfaftr)
                bfaftr = bfaftr != during
                flx = bfaftr.sum() / (2*dt_just_bfaftr)
                if sts.poisson(flx).logsf(nn/dt_mos) < plim_mos:
                    return True

            return False


        def search(xx, yy, tt):
            t0 = tt.min()
            t1 = tt.max()
            tr = t1 - t0

            dt = np.diff(tt)
            i = dt > 1e-6
            if i.sum()==0:
                return None

            dt = dt[i].min() * 10
            nf = int(tr/dt)
            if nf == 0:
                return None

            cr, bb = np.histogram(tt, nf)
            cs = convolve(cr.astype(np.double), np.ones(100)*1e-2, mode='reflect') + 1e-6
            pp = sts.poisson(cs).logsf(cr)

            p_poisson = pp < plim
            enough_counts = cr >= clim
            high_relative_amp = cr / cs > rlim

            for ii in np.where(p_poisson & enough_counts & high_relative_amp)[0]:
                i0 = (tt>=bb[ii]) & (tt<=bb[ii+1])
                nn = i0.sum()
                x0 = xx[i0]
                y0 = yy[i0]

                rr = np.column_stack((x0, y0))
                xg = np.linspace(x0.min(), x0.max(), 100)
                yg = np.linspace(y0.min(), y0.max(), 100)
                grid = np.array(np.meshgrid(xg, yg)).T.reshape(-1, 2)
                conv = gaussian_2d(rr, grid, sig).sum(axis=0).reshape(xg.size, yg.size)
                i1 = np.unravel_index(np.argmax(conv), conv.shape)
                c0 = conv[i1]
                # print(f'{c0:.3f}, {i0.sum():3d}, {c0/i0.sum():.3f}')

                if c0 > clim_conv and c0/nn > 0.25:
                    r0 = (xg[i1[0]], yg[i1[1]])
                    t = tt[i0]
                    t0 = t.min()
                    t1 = t.max()

                    if not check_mos(self, r0, t0, t1):
                        continue

                    obj = np.array([r0[0], r0[1], t0, t1, c0, nn, cr[ii], cs[ii]])
                    self.obj.append(obj)

                    if DEBUG:
                        print(f'{c0:.3f}, {i0.sum():3d}, {c0/i0.sum():.3f}')
                        plt.imshow(conv.T, interpolation='nearest', origin='lower')
                        i1 = (tt>=bb[np.max((ii-100, 0))]) & (tt<=bb[np.min((ii+101, bb.size-1))])
                        x1 = xx[i1]
                        y1 = yy[i1]
                        xm = x1.min()
                        ym = y1.min()
                        x1 = (x1 - xm) / 80
                        y1 = (y1 - ym) / 80
                        xb = int(x1.max())
                        yb = int(y1.max())
                        im, xb, yb = np.histogram2d(x1, y1, (xb, yb))
                        extent = np.array([xb[0], xb[-1], yb[0], yb[-1]])
                        plt.figure()
                        plt.imshow(im.T, interpolation='nearest', origin='lower', extent=extent, aspect='auto')
                        plt.plot((x0-xm)/80, (y0-ym)/80, 'or', ms=1)
                        plt.plot((r0[0]-xm)/80, (r0[1]-ym)/80, 'xg', ms=3)
                        plt.figure()
                        plt.plot(cr[ii-100:ii+101])
                        plt.plot(cs[ii-100:ii+101])
                        plt.grid()
                        plt.show()

        print('Finding sources')
        xx = self.xx_cam['PN']
        yy = self.yy_cam['PN']
        tt = self.tt_cam['PN']

        for ii in tqdm(range(12)):
            jj = self.ccd == ii+1
            if jj.sum() == 0:
                continue

            y0 = self.dy[jj].min()
            y1 = self.dy[jj].max()
            dy = int(np.ceil((y1 - y0)/3))

            for kk in range(3):
                ll = (self.dy >= y0 + kk*dy) & (self.dy <= y0 + (kk+1)*dy)
                mm = jj & ll
                if mm.sum() == 0:
                    continue

                search(xx[mm], yy[mm], tt[mm])


    def gen_products(self, figp):
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
                src.mk_postcard(figp)

        print('Generated {0:d} product(s)'.format(ctr))



class Source:
    def __init__(self, obs, obj):
        self.keys = obs.keys
        xx = obj[0]
        yy = obj[1]
        self.coo = np.array([xx, yy])
        self.t0 = obj[2]
        self.t1 = obj[3]
        ra, de = obs.wcs.all_pix2world(xx, yy, 1)
        self.sky = SkyCoord(ra*u.deg, de*u.deg)
        self.oi = obs.oi

        # Query catalogs
        self.nearest = np.empty(3)
        viz, self.plx_star, self.nearest[0] = get_viz(self.sky)
        sim, self.nearest[1], self.good_type, self.agn = get_sim(self.sky)
        # ned, self.nearest[2] = get_ned(self.sky)
        ned, self.nearest[2] = '', 999

        # Parse meta data
        buf = 100*'#' + '\nIndex ' + str(obs.ii) + '\n' + 100*'#' + '\n\n'
        buf = buf + 'Obs. Date: ' + obs.epoch + '\n'
        buf = buf + 'Obs. ID: ' + obs.oi + ', Target: ' + obs.target + '\n'
        buf = buf + 'RA, Dec:' + '{0:>16s}{1:>16s}'.format(*self.sky.to_string('hmsdms').split(' '))+ '\n'
        buf = buf + 'RA, Dec:' + '{0:>16s}{1:>16s}'.format(*self.sky.to_string('decimal').split(' ')) + '\n'
        buf = buf + 'l, b:   ' + '{0:>16s}{1:>16s}'.format(*self.sky.galactic.to_string('decimal').split(' ')) + '\n'

        # dc = (obj.max(0) - obj.min(0))
        # astrometry = np.sqrt(dc[0]**2+dc[1]**2) * 50e-3
        buf = buf + f'Test stats: {obj[4]:6.2f} (convolution), {obj[5]:3.0f} (counts), {obj[4]/obj[5]:6.2f} (frame source fraction)\n'
        buf = buf + f'Single-frame count: {obj[6]:6.2f}, Smoothed count rate: {obj[7]:6.2f} (counts per frame)\n'
        buf = buf + 'Exposure: {0:3.0f} ks, Time: {1:13.3f} s, Duration: {2:6.3f} s'.format((obs.tt.max()-obs.tt.min())/1.e3, self.t0, self.t1-self.t0) + '\n'

        self.viz = viz
        self.sim = sim
        self.ned = ned
        self.buf = buf
        self.src_rad = src_rad
        self.bkg_rad = [2*src_rad, 3*src_rad]
        self.backscal = self.src_rad**2/(self.bkg_rad[1]**2-self.bkg_rad[0]**2)

        during = (obs.tt > self.t0 - dt_mos/2) & (obs.tt < self.t1 + dt_mos/2)
        self.xd = obs.xx[during]
        self.yd = obs.yy[during]


    def keep(self, ctr):
        if self.good_type and not self.plx_star:
            self.out = '{0:s}_{1:03d}_bad'.format(self.oi, ctr+1)
            return True
        self.out = '{0:s}_{1:03d}_bad'.format(self.oi, ctr+1)
        return True


    def ext_evt(self, obs):
        def get_src_bkg_idx(self, tt, xx, yy):  # For spectra
            rad = (xx-self.coo[0])**2+(yy-self.coo[1])**2
            gd =  (tt >= self.t0-dt_mos) & (tt <= self.t1+dt_mos)
            src = (rad < self.src_rad**2) & gd
            bkg = (rad > self.bkg_rad[0]**2) & (rad < self.bkg_rad[1]**2) & gd
            return src, bkg

        def get_lc_idx(self, tt, xx, yy):  # For light curves
            rad = (xx-self.coo[0])**2+(yy-self.coo[1])**2
            during =  (tt >= self.t0-tpad) & (tt <= self.t1+tpad)
            src = (rad < self.src_rad**2)
            bkg = (rad > self.bkg_rad[0]**2) & (rad < self.bkg_rad[1]**2)
            return src & during, bkg & during, src, bkg

        def get_img_idx(self, tt, xx, yy):  # For images
            idx = (tt >= self.t0-dt_mos/2) & (tt <= self.t1+dt_mos/2)
            idx = idx & (xx > self.coo[0]-self.bkg_rad[1]) & (xx < self.coo[0]+self.bkg_rad[1]+1e-3)
            idx = idx & (yy > self.coo[1]-self.bkg_rad[1]) & (yy < self.coo[1]+self.bkg_rad[1]+1e-3)
            return idx

        # For and spectra
        src, bkg = get_src_bkg_idx(self, obs.tt, obs.xx, obs.yy)
        self.src_pi = obs.ee[src]
        self.bkg_pi = obs.ee[bkg]

        # For light curves
        src, bkg, src_llc, bkg_llc = get_lc_idx(self, obs.tt, obs.xx, obs.yy)
        self.src_tt = obs.tt[src]
        self.bkg_tt = obs.tt[bkg]
        self.src_llc = obs.tt[src_llc]
        self.bkg_llc = obs.tt[bkg_llc]

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

        # For observation image
        self.img_obs, self.xb, self.yb = obs.img, obs.xb, obs.yb


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

        t0 = np.min((self.src_llc.min(), self.bkg_llc.min()))
        t1 = np.max((self.src_llc.max(), self.bkg_llc.max()))
        self.tllc = np.linspace(t0, t1, 3*lc_nbin+1)

        self.src_llc, _ = np.histogram(self.src_llc, self.tllc)
        self.src_llc = self.src_llc/np.diff(self.tllc)
        self.bkg_llc, _ = np.histogram(self.bkg_llc, self.tllc)
        self.bkg_llc = self.backscal*self.bkg_llc/np.diff(self.tllc)
        self.sub_llc = self.src_llc - self.bkg_llc

    def mk_spec(self):
        self.src_spec, _ = np.histogram(self.src_pi, spec_bin)
        self.bkg_spec, _ = np.histogram(self.bkg_pi, spec_bin)
        self.src_spec = self.src_spec/np.diff(spec_bin)
        self.bkg_spec = self.backscal*self.bkg_spec/np.diff(spec_bin)

    def mk_postcard(self, figp):
        out_name = figp + self.out + '_pc.pdf'
        fig = plt.figure(figsize=(16, 16))

        # Light curve
        ax = plt.subplot2grid((4,3), (1,0), colspan=1, rowspan=1)
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
        ax.set_ylabel('Rate (s$^{-1}$)')
        ax.set_ylim(bottom=0)

        # Spectrum
        ax = plt.subplot2grid((4,3), (1,1), colspan=1, rowspan=1)
        tmp = np.empty(2*spec_bin.size-2)
        tmp[:-1:2] = spec_bin[:-1]
        tmp[1::2] = spec_bin[1:]
        plt.loglog(tmp, self.src_spec.repeat(2)+0.1, label='Source')
        plt.loglog(tmp, self.bkg_spec.repeat(2)+0.1, label='Background')
        ax.legend()
        ax.set_xlabel('Energy (keV)')
        ax.set_ylabel('Flux density (keV$^{-1}$)')

        # Zoomed image
        ax = plt.subplot2grid((4,3), (1,2), colspan=1, rowspan=1)
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
            ax = plt.subplot2grid((4,3), (ii//3+2, np.mod(ii,3)), colspan=1, rowspan=1)
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

        # Long light curve
        ax = plt.subplot2grid((4,3), (0,0), colspan=2, rowspan=1)
        tmp = np.empty(2*self.tllc.size-2)
        tmp[:-1:2] = self.tllc[:-1]
        tmp[1::2] = self.tllc[1:]
        ax.axvline(self.t0, c='gray')
        ax.axvline(self.t1, c='gray')
        plt.plot(tmp, self.src_llc.repeat(2), label='Source', color='greenyellow')
        plt.plot(tmp, self.bkg_llc.repeat(2), label='Background', color='gold')
        plt.plot(tmp, self.sub_llc.repeat(2), label='Subtracted', color='k')
        ax.legend()
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Rate (s$^{-1}$)')
        ax.set_ylim(bottom=0)

        # Observation image
        ax = plt.subplot2grid((4,3), (0,2), colspan=1, rowspan=1)
        plt.imshow(self.img_obs.T, cmap='afmhot', origin='lower')
        ix = np.digitize(self.coo[0], self.xb) - 1
        iy = np.digitize(self.coo[1], self.yb) - 1
        tmp = plt.Circle((ix, iy), radius=20, color='w', lw=1, fill=False)
        ax.add_artist(tmp)

        # Meta
        plt.text(0, 0, self.viz + self.sim + self.ned, fontdict={'family': 'monospace', 'size': 12}, transform=plt.gcf().transFigure, va='top')
        plt.text(0, 1, self.buf, fontdict={'family': 'monospace', 'size': 14}, transform=plt.gcf().transFigure, va='bottom')

        plt.savefig(out_name, bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.close()
        # np.savetxt(out_name.replace('_pc.pdf', '_lc.txt'), np.c_[self.tlc[:-1], self.src_lc, self.bkg_lc])
