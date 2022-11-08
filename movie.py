'''
2019-11-23, Dennis Alp, dalp@kth.se

Make movies of XMM observations.
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
import matplotlib.animation as animation
from matplotlib import cm
from matplotlib.colors import ListedColormap
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.wcs import WCS
from astropy.io import fits
from astropy import units
from astropy.time import Time


################################################################
#
class Observation:
    def __init__(self):
        print('Reading data')        
        self.tt = np.empty(0)
        self.ee = np.empty(0)
        self.xx = np.empty(0)
        self.yy = np.empty(0)
        
        for ff in glob(cwd + obs + '/*FTZ'):
            self.cam = ff.split('/')[-1][11:13]
            dd = fits.open(ff)[1].data

            if self.cam == 'PN':
                good = dd['FLAG'] <= XMMEA_EP
                good = good & (dd['PATTERN'] <= 4)
            elif self.cam == 'M1' or self.cam == 'M2':
                good = dd['FLAG'] <= XMMEA_EM
                good = good & (dd['PATTERN'] <= 12)
            else:
                print('ERROR Unknown instrument:', cam, obs)
                sys.exit(1)
                
            good = good & (dd['PI'] > ene_low*1.e3) & (dd['PI'] < ene_high*1.e3)
            if obs == '0655343801':
                good = good & (dd['X'] > 190000.) & (dd['X'] < 240000.)
                good = good & (dd['Y'] > 285000.) & (dd['Y'] < 330000.)
            else:
                good = good & (dd['X'] > 0.) # XMM puts some invalid photons at (-99999999.0, -99999999.0)
                good = good & (dd['Y'] > 0.)
                good = good & (dd['X'] < 100000.+dd['X'][good].mean())
                good = good & (dd['Y'] < 100000.+dd['Y'][good].mean())
            
            self.tt = np.concatenate((self.tt, dd['Time'][good]))
            self.ee = np.concatenate((self.ee, dd['PI'][good]/1.e3))
            self.xx = np.concatenate((self.xx, dd['X'][good]))
            self.yy = np.concatenate((self.yy, dd['Y'][good]))

    def mk_imgs(self):
        print('Making images')
        self.xbin = np.arange(self.xx.min(), self.xx.max(), sbin)
        self.ybin = np.arange(self.yy.min(), self.yy.max(), sbin)
        self.tbin = np.arange(self.tt.min(), self.tt.max(), tbin)
        bins = [self.xbin, self.ybin, self.tbin]
        self.imgs, _ = np.histogramdd(np.c_[self.xx, self.yy, self.tt], bins=bins)        


def mk_animation(imgs, tt):
    fig = plt.figure(figsize=(5.12, 5.12))
    ax = plt.gca()
    ax.axis('off')
    ax.set_aspect('equal')
    
    # Initial upper panels
    img = ax.imshow(np.zeros((imgs.shape[0], imgs.shape[1])), cmap='afmhot', interpolation='nearest', origin='lower')
#    axes[0,0].set_aspect('equal')
    global ann
    ann = ax.annotate(Time(tt[0], format='cxcsec').iso[:-7], (0, imgs.shape[1]), color='w', size=22, annotation_clip=True)

    def update_img(nn):
        print('Movie progress: {0:6.2f}%'.format(nn/imgs.shape[2]*100))
        
        # Annotate the year
        global ann
        ann.remove()
        ann = ax.annotate(Time(tt[nn], format='cxcsec').iso[:-7], (0, imgs.shape[1]), color='w', size=22, annotation_clip=True)
    
        # Plot the images
        # Upper right
        tmp = gaussian_filter(imgs[:,:,nn], 5).T
        img.set_data(tmp)
        img.set_clim([np.percentile(tmp, 60), np.percentile(tmp, 99.5)])
        return img

    ani = animation.FuncAnimation(fig, update_img, imgs.shape[2]-1)
    writer = animation.writers['ffmpeg'](fps=6)
    ani.save('/Users/silver/Desktop/' + obs + '.mp4', writer=writer, dpi=100, savefig_kwargs={'facecolor': 'k'})
    
   
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

    if os.path.isdir(cwd + obs):
        return True
    retcode = subprocess.call(['rm', '-rf', cwd + obs])
    retcode = subprocess.call(['rm'] + glob('GUEST*.tar'), stderr=subprocess.DEVNULL)
    retcode = subprocess.call(['rm'] + glob('*.FTZ'), stderr=subprocess.DEVNULL)

    tmp = dl_hlp(cmd + [purl.format(obs)])
    return dl_hlp(cmd + [murl.format(obs)]) or tmp



################################################################
# Just read the data
obs = sys.argv[1]
figp = '/Users/silver/Desktop/' + obs + '/fig/'
cwd = '/Users/silver/Desktop/' + obs + '/dat/'
retcode = subprocess.call(['mkdir', '-p', cwd])
retcode = subprocess.call(['mkdir', '-p', figp])
os.chdir(cwd)
cmd = ['curl', '-sS', '-O', '-J']
purl = 'http://nxsa.esac.esa.int/nxsa-sl/servlet/data-action-aio?level=PPS&extension=FTZ&name=PIEVLI&obsno={0:10s}'
murl = 'http://nxsa.esac.esa.int/nxsa-sl/servlet/data-action-aio?level=PPS&extension=FTZ&name=MIEVLI&obsno={0:10s}'
img_size = 1024 # pixel
binning = 80 # pixel, how much to bin the zoomed image (80 is default, results in a pixel size of 4 arcsec), i.e. pixel scale of 50 mas
sbin = 80 # spatial bin for source finding, pixel size of 20 arcsec
ene_low = 0.3 # keV
ene_high = 2.0 # keV
tbin = float(sys.argv[2]) # s
XMMEA_EP = 65584
XMMEA_EM = 65000

################################################################
# Abell 222: 0502020101
# Andromeda: 0112570101, 0600660301
print('Downloading data')
if download_data():
    observation = Observation()
    observation.mk_imgs()
    mk_animation(observation.imgs, observation.tbin)
#    retcode = subprocess.call(['rm', '-rf', cwd + obs])
else:
    print('WARNING No data found for Obs. ID:', obs, '(could be an RGS/OM only observation)')

#retcode = subprocess.call(['rm', '-rf', '/Users/silver/Desktop/' + obs + '/'])

#db()
