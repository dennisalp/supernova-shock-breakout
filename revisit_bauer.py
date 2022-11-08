'''
2020-12-07, Dennis Alp, dalp@kth.se

Reprocesses the objects rejected in the last step of the analysis to
search for Bauer-like objects. These would be extragalactic transients
but with counterparts not detected down to PanSTARRS. I parse the
coordinates from the .pdf postcards and use my general utility tool to
create new light curves and postcards with multiwavelength images.
'''

from __future__ import division, print_function
import os
from pdb import set_trace as db
import sys
from glob import glob
import time
from datetime import date
from datetime import timedelta
from tqdm import tqdm
import subprocess

import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.io import fits
from astropy import units
from astropy import units as u
from astropy.time import Time

from astroquery.vizier import Vizier
from astroquery.simbad import Simbad
from astroquery.ned import Ned
from tika import parser

import scipy.stats as sts
from scipy.ndimage import zoom
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata
import matplotlib.gridspec as gs


# Constants, cgs
cc = 2.99792458e10 # cm s-1
GG = 6.67259e-8 # cm3 g-1 s-2
hh = 6.6260755e-27 # erg s
DD = 51.2 # kpc
pc = 3.086e18 # cm
kpc = 3.086e21 # cm
mpc = 3.086e24 # cm
kev2erg = 1.60218e-9 # erg keV-1
kev2hz = kev2erg/hh
hz2kev = hh/kev2erg
Msun = 1.989e33 # g
Lsun = 3.828e33 # erg s-1
Rsun = 6.957e10 # cm
Tsun = 5772 # K
uu = 1.660539040e-24 # g
SBc = 5.670367e-5 # erg cm-2 K-4 s-1
kB = 1.38064852e-16 # erg K-1
mp = 1.67262192369e-24 # g


files = sorted(glob('/Users/silver/box/phd/pro/sne/sbo/revisit_bauer/inp/*.pdf'))
nn = len(files)
coo = []
obs = []
num = []

for ff in tqdm(files):
    raw = parser.from_file(ff)
    # print(list(raw.keys()))
    # ['metadata', 'status', 'content']
    
    for ll in raw['content'].split('\n'):
        if 'RA, Dec:       ' in ll:
            ll = ll.split()
            coo.append(ll[2] + ' ' + ll[3])
            obs.append(ff.split('/')[-1].split('_')[0])
            num.append(ff.split('/')[-1].split('_')[1][:-4])


coo = SkyCoord(coo, unit=(units.deg, units.deg))
obs = np.array(obs)
num = np.array(num)







################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################






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
        self.xx_cam = {'PN': np.empty(0), 'M1': np.empty(0), 'M2': np.empty(0)}
        self.yy_cam = {'PN': np.empty(0), 'M1': np.empty(0), 'M2': np.empty(0)}
        self.keys = ['PN', 'M1', 'M2']

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
            good = good & (dd['X'] > 0.) # XMM puts some invalid photons at (-99999999.0, -99999999.0)
            good = good & (dd['Y'] > 0.)

            self.xx_cam[self.cam] = np.concatenate((self.xx_cam[self.cam], dd['X'][good]))
            self.yy_cam[self.cam] = np.concatenate((self.yy_cam[self.cam], dd['Y'][good]))

        for kk in self.keys:
            self.xx = np.concatenate((self.xx, self.xx_cam[kk]))
            self.yy = np.concatenate((self.yy, self.yy_cam[kk]))

        self.wcs = get_wcs(ff)
        self.img, self.xbin, self.ybin = np.histogram2d(self.xx, self.yy, img_size)


        
class Xoi:
    def __init__(self, ii):
        self.keys = ['PN', 'M1', 'M2']
        # Query catalogs
        self.nearest = np.empty(3)
        viz, self.plx_star, self.nearest[0] = get_viz(coo[ii])
        sim, self.nearest[1], self.good_type, self.agn = get_sim(coo[ii])
        ned, self.nearest[2] = get_ned(coo[ii])

        # Parse meta data
        buf = 100*'#' + '\nIndex ' + str(ii) + '\n' + 100*'#' + '\n\n'
        buf = buf + 'Obs. ID: ' + obs[ii] + '\n'
        buf = buf + 'RA, Dec:' + '{0:>16s}{1:>16s}'.format(*coo[ii].to_string('hmsdms').split(' '))+ '\n'
        buf = buf + 'RA, Dec:' + '{0:>16s}{1:>16s}'.format(*coo[ii].to_string('decimal').split(' ')) + '\n'
        buf = buf + 'l, b:   ' + '{0:>16s}{1:>16s}'.format(*coo[ii].galactic.to_string('decimal').split(' ')) + '\n\n'

        self.ii = ii
        self.out = obs_num.format(obs[ii], num[ii])
        self.viz = viz
        self.sim = sim
        self.ned = ned
        self.buf = buf
        self.src_tt = np.empty(0)
        self.src_pi = np.empty(0)
        self.bkg_tt = np.empty(0)
        self.bkg_pi = np.empty(0)
        self.src_rad = 400 # pixels, pixel size of 50 mas
        self.bkg_rad = [1000, 2000]
        self.backscal = self.src_rad**2/(self.bkg_rad[1]**2-self.bkg_rad[0]**2)

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

        
        self.img_cam = {}
        for kk in self.keys:
            self.img_cam[kk], _, _ = np.histogram2d(observation.xx_cam[kk], observation.yy_cam[kk], [xbin, ybin])
        
    def mk_lc(self):
        self.tlc = self.src_tt[::cts_per_bin]
        self.src_lc = cts_per_bin/np.diff(self.tlc)
        self.bkg_lc, _ = np.histogram(self.bkg_tt, self.tlc)
        self.bkg_lc = self.backscal*self.bkg_lc/np.diff(self.tlc)
        self.sub_lc = self.src_lc - self.bkg_lc
        
    def mk_spec(self):
        self.src_spec, _ = np.histogram(self.src_pi, spec_bin)
        self.bkg_spec, _ = np.histogram(self.bkg_pi, spec_bin)
        self.src_spec = self.src_spec/np.diff(spec_bin)
        self.bkg_spec = self.backscal*self.bkg_spec/np.diff(spec_bin)

    def mk_postcard(self):
        out_name = self.out + '.pdf'
        fig = plt.figure(figsize=(20, 20))
        gs.GridSpec(4,4)
#        fig.subplots_adjust(hspace = 0.5)

        # Light curve
        ax = plt.subplot2grid((4,4), (0,0), colspan=2, rowspan=1)
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
        ax = plt.subplot2grid((4,4), (0,2), colspan=1, rowspan=1)
        tmp = np.empty(2*spec_bin.size-2)
        tmp[:-1:2] = spec_bin[:-1]
        tmp[1::2] = spec_bin[1:]
        plt.loglog(tmp, self.src_spec.repeat(2), label='Source')
        plt.loglog(tmp, self.bkg_spec.repeat(2), label='Background')
        ax.legend()
        ax.set_xlabel('Energy (keV)'.format(shift))
        ax.set_ylabel('Flux density (cts/keV)')
        
        # Zoomed image
        ax = plt.subplot2grid((4,4), (1,0), colspan=1, rowspan=1)
        plt.imshow(self.img.T, cmap='afmhot', origin='lower')
        tmp = plt.Circle((self.bkg_rad[1]/binning-0.5, self.bkg_rad[1]/binning-0.5), self.src_rad/binning, color='w', lw=2, fill=False)
        ax.add_artist(tmp)
        tmp = plt.Circle((self.bkg_rad[1]/binning-0.5, self.bkg_rad[1]/binning-0.5), self.bkg_rad[0]/binning, color='w', lw=1, fill=False)
        ax.add_artist(tmp)
        tmp = plt.Circle((self.bkg_rad[1]/binning-0.5, self.bkg_rad[1]/binning-0.5), self.bkg_rad[1]/binning, color='w', lw=1, fill=False)
        ax.add_artist(tmp)
        ax.set_axis_off()

        # Camera image
        for ii in range(len(self.keys)):
            kk = self.keys[np.mod(ii,len(self.img_cam.keys()))]
            ax = plt.subplot2grid((4,4), (1, ii+1), colspan=1, rowspan=1)
            tmp = self.img_cam[kk].T
            plt.imshow(tmp, cmap='afmhot', origin='lower')
            tmp = plt.Circle((self.bkg_rad[1]/binning-0.5, self.bkg_rad[1]/binning-0.5), self.src_rad/binning, color='w', lw=2, fill=False)
            ax.add_artist(tmp)
            tmp = plt.Circle((self.bkg_rad[1]/binning-0.5, self.bkg_rad[1]/binning-0.5), self.bkg_rad[0]/binning, color='w', lw=1, fill=False)
            ax.add_artist(tmp)
            tmp = plt.Circle((self.bkg_rad[1]/binning-0.5, self.bkg_rad[1]/binning-0.5), self.bkg_rad[1]/binning, color='w', lw=1, fill=False)
            ax.add_artist(tmp)
            ax.set_axis_off()

        # Image
        ax = plt.subplot2grid((4,4), (0,3), colspan=1, rowspan=1)
        tmp = np.log10(observation.img+1).T
        tmp = gaussian_filter(tmp,2)
        plt.imshow(tmp, vmin=np.percentile(tmp, 60.), vmax=np.percentile(tmp, 99.9), cmap='afmhot', origin='lower')

        tmp = float(griddata(observation.xbin, np.arange(0,img_size+1), self.coo[0]))-0.5
        tmp = [tmp, float(griddata(observation.ybin, np.arange(0,img_size+1), self.coo[1]))-0.5]
        tmp = plt.Circle(tmp, 40, color='greenyellow', lw=2, fill=False)
        ax.add_artist(tmp)
        ax.set_axis_off()

        # HiPS images
        for kk, cat in enumerate(cats):
            ax = plt.subplot2grid((4,4), (2+kk//4, np.mod(kk,4)), colspan=1, rowspan=1)
            plt_hips(fov[kk], ax, cat, coo[self.ii], lab[kk])

        # Meta
        plt.text(0, 0, self.viz + self.sim + self.ned, fontdict={'family': 'monospace', 'size': 12}, transform=plt.gcf().transFigure, va='top')
        plt.text(0, 1, self.buf, fontdict={'family': 'monospace', 'size': 14}, transform=plt.gcf().transFigure, va='bottom')
        
        plt.savefig(out_name, bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.close()
        return 1



# https://aladin.u-strasbg.fr/hips/list
def plt_hips(fov, ax, cat, coo, lab):
    ra, de = coo.ra.deg, coo.dec.deg
    url = hips.format(cat, hips_width, hips_height, fov, ra, de)
    try:
        url = fits.open(url)
        if 'XRT' in cat:
            img = url[0].data
            img = np.log10(img+img.max()/100.)
            img = img/img.max()
        elif 'HSC/DR2/wide/r' in cat:
            img = url[0].data
            norm = img.max()
            if np.abs(norm) > 1.e-12:
                img = img/norm
        else:
            img = np.zeros((hips_width, hips_height, 3))
            img[:,:,0] = url[0].data[0]
            img[:,:,1] = url[0].data[1]
            img[:,:,2] = url[0].data[2]
    
            norm = img.max(axis=(0,1))
            if np.min(np.abs(norm)) > 1.e-12:
                img = img/norm
    except:
        img = np.ones((hips_width, hips_height))
    
    ax.imshow(img, origin='lower', interpolation='nearest')
    ax.set_title(lab)
    ax.plot()
    xx = hips_width/2
    xs = hips_width/10
    ax.plot([xx+0.2*xs, xx+1.2*xs], [xx, xx], color=contrast_col, lw=0.8)
    ax.plot([xx, xx], [xx+0.2*xs, xx+1.2*xs], color=contrast_col, lw=0.8)

    ax.set_axis_off()

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

#    print('Downloading data for Obs. ID', obs[ii], '(Index: {0:d})'.format(ii))
    retcode = subprocess.call(['rm', '-rf', cwd + obs[ii]])
    retcode = subprocess.call(['rm'] + glob('GUEST*.tar'), stderr=subprocess.DEVNULL)
    retcode = subprocess.call(['rm'] + glob('*.FTZ'), stderr=subprocess.DEVNULL)

    tmp = dl_hlp(cmd + [purl.format(obs[ii])])
    return dl_hlp(cmd + [murl.format(obs[ii])]) or tmp




################################################################
# Just read the data
cwd = '/Users/silver/box/phd/pro/sne/sbo/revisit_bauer/'
os.chdir(cwd)
cmd = ['curl', '-sS', '-O', '-J']
obs_num = '{0:10s}_{1:3s}'
purl = 'http://nxsa.esac.esa.int/nxsa-sl/servlet/data-action-aio?level=PPS&extension=FTZ&name=PIEVLI&obsno={0:10s}'
murl = 'http://nxsa.esac.esa.int/nxsa-sl/servlet/data-action-aio?level=PPS&extension=FTZ&name=MIEVLI&obsno={0:10s}'
img_size = 1024 # pixel
binning = 80 # pixel, how much to bin the zoomed image (80 is default, results in a pixel size of 4 arcsec)
cts_per_bin = 25 # For the light curves
spec_bin = np.logspace(-1,1,20) # keV
XMMEA_EP = 65584
XMMEA_EM = 65000

contrast_col = '#ff0000'
hips = 'http://alasky.u-strasbg.fr/hips-image-services/hips2fits?hips={}&width={}&height={}&fov={}&projection=TAN&coordsys=icrs&ra={}&dec={}'
hips_width = 256
hips_height = 256
fov = np.array([0.3, 0.3 , 0.3, 0.3, 0.3, 0.3, 0.3, 0.3])/60

# https://aladin.u-strasbg.fr/hips/list
cats = ['CDS/P/DSS2/color',
        'nasa.heasarc/P/Swift/XRT/exp', # 'CDS/P/SDSS9/color-alt'
        'cxc.harvard.edu/P/cda/hips/allsky/rgb',
        'CDS/P/2MASS/color',
        'CDS/P/PanSTARRS/DR1/color-z-zg-g',
        'CDS/P/DECaLS/DR5/color',
        'cds/P/DES-DR1/ColorIRG',
        'CDS/P/HSC/DR2/wide/r',]
lab = ['DSS', 'XRT', 'CXO', '2MASS', 'PS1', 'DECaLS', 'DES', 'HSC']


################################################################
for ii in range(0, obs.size):
    xoi = Xoi(ii)
    download_data(ii)
    observation = Observation(ii)

    xoi.ext_evt(observation.wcs)
    xoi.mk_img(observation)
    xoi.mk_lc()
    xoi.mk_spec()
    xoi.mk_postcard()

    retcode = subprocess.call(['rm', '-rf', cwd + obs[ii]])
    del(observation)



# db()
