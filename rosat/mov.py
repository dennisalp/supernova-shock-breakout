'''
2021-05-28, Dennis Alp, dalp@kth.se

Make movies of ROSAT observations.
'''

from __future__ import division, print_function
import os
from pdb import set_trace as db
import sys
import time
from glob import glob
from datetime import timedelta
from datetime import datetime
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
def pi2kev(pi):
    return pi*0.01
def kev2pi(kev):
    return kev/0.01

class Observation:
    def __init__(self):
        print('Reading data')        
        self.tt = np.empty(0)
        self.ee = np.empty(0)
        self.xx = np.empty(0)
        self.yy = np.empty(0)

        dd = fits.open(obs + '_bc.fits')[2].data
    
        good = (dd['PI'] >= kev2pi(ene_low)) & (dd['PI'] <= kev2pi(ene_high))
                
        self.tt = dd['TIME'][good]
        self.t2 = self.tt.copy()
        dt = np.diff(self.tt)
        for tt in np.where(dt>100)[0]:
            self.t2[tt+1:] = self.t2[tt+1:]-dt[tt]
            
        self.ee = dd['PI'][good]
        self.xx = dd['X'][good]
        self.yy = dd['Y'][good]
    
    def mk_imgs(self):
        print('Making images')
        self.xbin = np.arange(self.xx.min(), self.xx.max(), sbin)
        self.ybin = np.arange(self.yy.min(), self.yy.max(), sbin)
        self.tbin = np.arange(self.t2.min(), self.t2.max(), tbin)
        bins = [self.xbin, self.ybin, self.tbin]
        self.imgs, _ = np.histogramdd(np.c_[self.xx, self.yy, self.t2], bins=bins)        


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
        tmp = launch + timedelta(0,tt[nn]) # days, seconds, then other fields.
        ann = ax.annotate(str(tmp)[:19], (0, imgs.shape[1]), color='w', size=22, annotation_clip=True)
    
        # Plot the images
        # Upper right
        tmp = gaussian_filter(imgs[:,:,nn], 1).T
        img.set_data(tmp)
        img.set_clim([vmin, vmax])
        return img

    ani = animation.FuncAnimation(fig, update_img, imgs.shape[2]-1)
    # vmin, vmax = np.percentile(imgs, 60), np.percentile(imgs, 99.5)
    vmin, vmax = 0., imgs.max()/20.
    writer = animation.writers['ffmpeg'](fps=6)
    ani.save('/Users/silver/Desktop/' + obs + '.mp4', writer=writer, dpi=100, savefig_kwargs={'facecolor': 'k'})


################################################################
# Just read the data
cwd = '/Users/silver/out/ros/cat/'
os.chdir(cwd)

obs = sys.argv[1]
sbin = 10 # spatial bin for source finding, pixel size of 20 arcsec
ene_low = 0.1 # keV
ene_high = 2.0 # keV
tbin = float(sys.argv[2]) # s
launch = datetime(1990,6,1,21,6,50) # http://hea-www.harvard.edu/PROS/PUG/node28.html

################################################################
observation = Observation()
observation.mk_imgs()
mk_animation(observation.imgs, observation.tbin)
