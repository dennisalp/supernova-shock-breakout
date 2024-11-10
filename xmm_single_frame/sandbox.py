from pdb import set_trace as st
import os
import subprocess
from glob import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from astroquery.esa.xmm_newton import XMMNewton
from astropy.io import fits



def plot_pps(oi, inst, dt, good=True):
    fn = f'{oi}_{inst}.FTZ'
    df = fits.open(fn)[1].data

    x0 = np.mean(df.X)
    y0 = np.mean(df.Y)
    arcmin = 5
    r0 = 20 * 60 * arcmin  # pixel = pixel arcsec-1 * arcsec arcmin-1 * arcmin (number of pixels to get 5 arcmin half-side)
    bins = 2*arcmin*15  # number of bins to get 4 arcsec bin-1
    ii = (df.X > x0-r0) & (df.X < x0+r0) & (df.Y > y0-r0) & (df.Y < y0+r0)

    if good:
        ii = ii & (df.PI > .3*1.e3)
        ii = ii & (df.PI < 10*1.e3)
        if inst == 'PN':
            ii = ii & (df.FLAG <= XMMEA_EP)
            ii = ii & (df.PATTERN <= 4)
        elif inst == 'M1' or inst == 'M2':
            ii = ii & (df.FLAG <= XMMEA_EM)
            ii = ii & (df.PATTERN <= 12)
        gb = 'gd'
    else:
        gb = 'bd'

    nn = ii.sum()
    rate = nn / (df.TIME[ii].max() - df.TIME[ii].min()) * dt/1000
    title = f"{fn.replace('.FTZ', '')}_{gb} events per frame: {rate:.4f} ({nn} total events within $10^2$ arcmin$^2$)"
    fn = "../fig/" + fn.replace('.FTZ', f'_{gb}.pdf')
    print(title)

    im, xb, yb = np.histogram2d(df.X[ii], df.Y[ii], bins=bins)
    dx = xb[1] - xb[0]
    dy = yb[1] - yb[0]
    dx = 10 / (50 * dx * 1e-3)  # 10 arcsec / (50 mas pixel-1 * pixel bin-1 * 1e-3 arcsec mas-1)
    dy = 10 / (50 * dy * 1e-3)
    lg = np.log10(im+1)

    plt.imshow(im, origin='lower', cmap='hot', interpolation='nearest')
    ellipse = patches.Ellipse((50, 50), width=2*dx, height=2*dy, fill=False, edgecolor='white', linewidth=1)
    plt.gca().add_patch(ellipse)
    plt.colorbar(label='Counts')
    plt.xlabel('X (4 arcsec)')
    plt.ylabel('Y (4 arcsec)')
    plt.title(title)
    plt.savefig(fn, bbox_inches='tight', dpi=600)
    # plt.show()
    plt.close()


def plot_odf(oi, inst, dt):
    fn = max(glob(f'{oi}/*{inst}*IME.FIT'), key=os.path.getsize)
    df = fits.open(fn)[1].data

    x0 = np.mean(df.RAWX)
    y0 = np.mean(df.RAWY)

    fn = fn.split('/')[-1]
    title = f"{oi}_{inst}_rw 'events' on CCD {df.shape[0]}"
    fn = f"../fig/{oi}_{inst}_rw.pdf"
    label = 'Threshold-crossing Pixels (not patterns/events)' if inst == 'PN' else 'ODF Events'

    bx = df.RAWX.max() - df.RAWX.min()
    by = df.RAWY.max() - df.RAWY.min()
    im, xb, yb = np.histogram2d(df.RAWX, df.RAWY, bins=(bx, by))
    dx = xb[1] - xb[0]
    dy = yb[1] - yb[0]

    res = 4.1 if inst == 'PN' else 1.1
    dx = 10 / (res * dx)  # 10 arcsec / (50 mas pixel-1 * pixel bin-1 * 1e-3 arcsec mas-1)
    dy = 10 / (res * dy)

    plt.imshow(im, origin='lower', cmap='hot', interpolation='nearest', vmax=np.percentile(im, 99.9))
    ellipse = patches.Ellipse((10, 10), width=2*dx, height=2*dy, fill=False, edgecolor='white', linewidth=1)
    plt.gca().add_patch(ellipse)
    plt.colorbar(label=label)
    plt.xlabel(f'X ({res}ish arcsec)')
    plt.ylabel(f'Y ({res}ish arcsec)')
    plt.title(title)
    plt.savefig(fn, bbox_inches='tight', dpi=600)
    # plt.show()
    plt.close()



exps = [
    ('0111360101', 'PN',   73),  # on-axis, Cyg X-2, pn Full Frame, 73 ms
    ('0103261901', 'M1', 2600),  # off-axis, Prime Full Frame, 4U 1758-25 (Low Mass X-ray Binary)
    ('0103261901', 'M2', 2600),
    ('0103261901', 'PN',   73),
    ('0411082701', 'PN',   48),  # on-axis, Mkn 421, pn Large Window, 48 ms
    ('0122340601', 'M1', 2600),  # off-axis, Prime Full Frame, GX13+1 off5 (Low Mass X-ray Binary)
    ('0122340601', 'M2', 2600),
    ('0122340601', 'PN',   73),
    ('0823990901', 'PN',   73),  # off-axis, Full Frame, X Nor X-1 (Low Mass X-ray Binary)
]

cwd = '/Users/silver/xmm/sandbox/dat/'
retcode = subprocess.call(['mkdir', '-p', cwd])
os.chdir(cwd)

# Concerning flags
# https://xmm-tools.cosmos.esa.int/external/sas/current/doc/eimageget/node4.html
# http://xmm.esac.esa.int/xmmhelp/EPICpn?id=8560;expression=xmmea;user=guest
XMMEA_EP = 65584
XMMEA_EM = 65000

# for oi, inst, dt in exps:
#     fn = f'{oi}_{inst}.FTZ'
#     name = 'PIEVLI' if inst == 'PN' else 'MIEVLI'

#     if not os.path.exists(fn):
#         XMMNewton.download_data(oi, level="PPS", instname=inst, name=name, filename=fn, extension='FTZ', verbose=False)

#     plot_pps(oi, inst, dt, True)
#     plot_pps(oi, inst, dt, False)

for oi, inst, dt in exps:
    if not os.path.isdir(oi):
        fn = f'{oi}.tar.gz'
        XMMNewton.download_data(oi, level="ODF", verbose=False)
        retcode = subprocess.call(['mkdir', '-p', oi])
        retcode = subprocess.call(['tar', '-xvf', fn, '-C', oi], stderr=subprocess.DEVNULL)
        tar = glob(f'{oi}/*.TAR')[0]
        retcode = subprocess.call(['tar', '-xvf', tar, '-C', oi], stderr=subprocess.DEVNULL)

    plot_odf(oi, inst, dt)

st()
