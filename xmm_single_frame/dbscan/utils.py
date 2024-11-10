from pdb import set_trace as st
from itertools import product, combinations
import pickle

import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.wcs import WCS

plt.switch_backend('agg')



def plt_obs_img(figp, xx, yy):
    img = np.histogram2d(xx, yy, 512)[0]
    img = np.log10(img+1)
    plt.imshow(img, interpolation='nearest')
    plt.title(f'{xx.shape[0]} events')
    plt.savefig(figp, bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close()


def scatter3d(xyt, nn=100):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(xyt[:nn,0], xyt[:nn,1], xyt[:nn,2])
    plt.show()


def scatter3d_obj_hlp(xyt, mmp, cc, ic):
    x0 = cc[:, 0].min()
    y0 = cc[:, 1].min()
    t0 = cc[:, 2].min()
    x1 = cc[:, 0].max()
    y1 = cc[:, 1].max()
    t1 = cc[:, 2].max()

    # Plot edges of the cube spanned by cc
    rr = [
        [x0, x1],  # X range
        [y0, y1],  # Y range
        [t0, t1]   # Z range (Time range)
    ]

    ii = (mmp[:, 0] > x0 - 2) & (mmp[:, 0] < x1 + 2)
    ii = (mmp[:, 1] > y0 - 2) & (mmp[:, 1] < y1 + 2) & ii
    ii = (mmp[:, 2] > t0 - 10) & (mmp[:, 2] < t1 + 10) & ii

    scatter3d = {}
    scatter3d['rr'] = rr
    scatter3d['ncp'] = (mmp[ii, 0], mmp[ii, 1], mmp[ii, 2])
    scatter3d['cp'] = (xyt[ic, 0], xyt[ic, 1], xyt[ic, 2])

    return scatter3d


def scatter3d_obj_plt(scatter3d, figp=None):
    rr = scatter3d['rr']
    ncp = scatter3d['ncp']
    cp = scatter3d['cp']

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(ncp[0], ncp[1], ncp[2], label='Non-cluster points')
    ax.scatter(cp[0], cp[1], cp[2], label='Cluster points')

    for s, e in combinations(np.array(list(product(*rr))), 2):
        if np.sum(np.abs(s - e)) in (rr[0][1] - rr[0][0], rr[1][1] - rr[1][0], rr[2][1] - rr[2][0]):
            ax.plot3D(*zip(s, e), color="r")

    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Time Axis')
    ax.legend()

    if figp:
        plt.savefig(figp, bbox_inches='tight', pad_inches=0.1, dpi=300)
        pklp = figp.replace('.pdf', '.pkl')
        with open(pklp, 'wb') as f:
            pickle.dump(fig, f)
        plt.close()
    else:
        plt.show()


def scatter3d_obj_load(pklp):
    with open(pklp, 'rb') as f:
        fig = pickle.load(f)
    plt.show()


def get_wcs(ff):
    dd = fits.open(ff)
    wcs = WCS()
    wcs.wcs.crpix = [dd[0].header['REFXCRPX'], dd[0].header['REFYCRPX']]
    wcs.wcs.cdelt = [dd[0].header['REFXCDLT'], dd[0].header['REFYCDLT']]
    wcs.wcs.crval = [dd[0].header['REFXCRVL'], dd[0].header['REFYCRVL']]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return wcs
