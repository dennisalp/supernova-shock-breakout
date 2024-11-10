from pdb import set_trace as st
from itertools import product, combinations
import pickle

import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.wcs import WCS

plt.switch_backend('agg')



def plt_obs_img(figp, xx, yy):
    img, xb, yb = np.histogram2d(xx, yy, 512)
    img = np.log10(img+1)
    plt.imshow(img.T, interpolation='nearest', cmap='afmhot', origin='lower')
    plt.title(f'{xx.shape[0]} events')
    plt.savefig(figp, bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close()
    return img, xb, yb


def get_wcs(ff):
    dd = fits.open(ff)
    wcs = WCS()
    wcs.wcs.crpix = [dd[0].header['REFXCRPX'], dd[0].header['REFYCRPX']]
    wcs.wcs.cdelt = [dd[0].header['REFXCDLT'], dd[0].header['REFYCDLT']]
    wcs.wcs.crval = [dd[0].header['REFXCRVL'], dd[0].header['REFYCRVL']]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return wcs
