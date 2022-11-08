'''
2019-10-16, Dennis Alp, dalp@kth.se

Search the the custom light curves to find transients.
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
from astroquery.vizier import Vizier
from astroquery.simbad import Simbad
from astroquery.ned import Ned
from astropy.time import Time




################################################################
# Help functions
################################################################
def plt_lc():
    tmp = np.empty(2*tt.size-2)
    tmp[:-1:2] = tt[:-1]
    tmp[1::2] = tt[1:]

    fig = plt.figure(figsize=(14, 7.5))
    ax = plt.gca()
    plt.plot(tmp, src.repeat(2), label='Source', color='greenyellow')
    plt.plot(tmp, bkg.repeat(2), label='Background', color='gold')
    plt.plot(tmp, sub.repeat(2), label='Subtracted', color='k')
    plt.errorbar((tt[:-1]+tt[1:])/2., sub, yerr=src/np.sqrt(cts_per_bin), color='k', fmt='.', ms=0)
    ax.legend()
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Rate (cts/s)')
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0, right=tmp.max())
    plt.show()

def my_percentile(val, per, wei):
    idx = np.argsort(val)
    val = val[idx]
    wei = wei[idx]

    per_wei = (np.cumsum(wei)-.5*wei)/wei.sum()
    return np.interp(per, per_wei, val)

def mk_note(ff):
    if not ff in notes:
        subprocess.call(['touch', 'discarded/' + ff])




################################################################
# Heuristics
################################################################
def fea90p(lim):
    score = sub.max()/np.abs(my_percentile(sub, np.array([0.9]), np.diff(tt))[0])
    if score > lim:
        print('fea90p', cts+1, ii, ff, score)
        plt_lc()
        return 1
    return 0

def fea70p(lim):
    score = sub.max()/np.abs(my_percentile(sub, np.array([0.7]), np.diff(tt))[0])
    if score > lim:
        print('fea70p', cts+1, ii, ff, score)
        plt_lc()
        return 1
    return 0

def fea50p_sb(lim, sb_lim):
    score = sub/np.abs(my_percentile(sub, np.array([0.5]), np.diff(tt))[0])
    sb = np.where(bkg < 1e-10, 99999999, src/bkg)
    condition = (score > lim) & (sb > sb_lim)
    if np.any(condition):
        print('fea50p_sb', cts+1, ii, ff, condition)
        plt_lc()
        return 1
    return 0

def top_hat(width, upp, sb_lim, low_time):
    ker = np.ones(width)
    sig = src/np.sqrt(cts_per_bin)
    kk = np.abs(sub/sig) < 1
    smo = np.convolve(src, ker, 'same')
    sb = np.where(bkg < 1e-10, 99999999, smo/np.convolve(bkg, ker, 'same'))

    if kk.sum() == 0 or (~kk).sum() == 0: return 0
    tot = np.sum(np.diff(tt)[kk])
    condition = (sb[~kk] > sb_lim) & (smo[~kk] > upp)
    if tot > low_time and np.any(condition):
        print('top_hat', cts+1, ii, ff, sb[~kk].max())
        plt_lc()
        return 1
    return 0        
    

def quiet_flare(upp, sb_lim, low_time):
    sb = np.where(bkg < 1e-10, 99999999, src/bkg)
    sig = src/np.sqrt(cts_per_bin)
    kk = np.abs(sub/sig) < 1

    if kk.sum() == 0 or (~kk).sum() == 0: return 0
    tot = np.sum(np.diff(tt)[kk])
    condition = (src[~kk] > upp) & (sb[~kk] > sb_lim)
    if tot > low_time and np.any(condition):
        ll = np.argmax(condition)
        print('quiet_flare', cts+1, ii, ff, src[~kk][ll], sb[~kk][ll], tot)
        plt_lc()
        return 1
    return 0



################################################################
# Paramters and preparation
################################################################
cts_per_bin = 25 # For the light curves
cwd = '/Users/silver/Box Sync/phd/pro/sne/sbo/inspection/'
os.chdir(cwd)
files = sorted(glob('/Volumes/pow/out/sne/sbo/constant/*/*/*/*.txt'))[:]

notes = set()
for ff in glob('**/*.txt', recursive=True):
    notes.add(ff.split('/')[-1])
pdfs = set()
for ff in glob('**/*.pdf', recursive=True):
    pdfs.add(ff.split('/')[-1])



    
################################################################
# Main loop
################################################################
cts = 0
for ii, ff in enumerate(files):
    print(ii)
    # Skip items with existing notes file
    if ff.split('/')[-1] in notes:
        continue
    # Skip items with existing pdf file (and create a notes file)
    elif ff.split('/')[-1].replace('.txt', '.pdf') in pdfs:
        mk_note(ff.split('/')[-1])
        continue



    # Parse my LC files
    lc = np.loadtxt(ff)
    src = lc[:,0]
    bkg = lc[:,1]
    sub = lc[:,2]
    tt = np.cumsum(cts_per_bin/src)
    tt = np.insert(tt, 0, 0.)



    # Look at things that have not been selected
    # if np.random.rand() < 0.01:
    #     plt_lc()
    #     continue
    
    
    
    # Cut background
    jj = bkg < 0.05
    src = src[jj]
    bkg = bkg[jj]
    sub = sub[jj]
    tt = np.cumsum(cts_per_bin/src)
    tt = np.insert(tt, 0, 0.)
    if jj.sum() < 4:
        continue


    ret = fea50p_sb(3, 10)
    if ret == 0: ret = fea50p_sb(5, 3)
    if ret == 0: ret = top_hat(3, 0.1, 3, 10000)
    if ret == 0: ret = quiet_flare(0.05, 3, 10000)
    
    
    
    if ret:
        cts += 1
        mk_note(ff.split('/')[-1])


# db()
