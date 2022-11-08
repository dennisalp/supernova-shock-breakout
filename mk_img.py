'''
2020-02-06, Dennis Alp, dalp@kth.se

Get sky images from HiPS and overlay XMM contours.
'''

from __future__ import division, print_function
import os
from pdb import set_trace as db
import sys
from glob import glob
import time
from datetime import date
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy import misc
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy import units as u
from astropy.io import fits
from astropy import units
from astropy import utils as apu

from astroquery.simbad import Simbad
# from astropy.visualization import astropy_mpl_style
# plt.style.use(astropy_mpl_style)
from urllib.parse import quote
from astropy.visualization import (MinMaxInterval, SqrtStretch, AsinhStretch, ImageNormalize)

#For LaTeX style font in plots
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rc('text', usetex=True)

apu.data.Conf.remote_timeout.set(value=60.)

################################################################
# Help functions
def get_wcs(hd):
    wcs = WCS()
    wcs.wcs.crpix = [hd['CRPIX1'], hd['CRPIX2']]
    wcs.wcs.cdelt = [hd['CDELT1'], hd['CDELT2']]
    wcs.wcs.crval = [hd['CRVAL1'], hd['CRVAL2']]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return wcs

def mk_cross(xx, yy):
    rr = 3 # arcsec
    rr = rr*width/(fov*3600)
    if xx < 0 and yy < 0:
        xs = -1
        ys = -1
    elif xx < 0 and yy > 0:
        xs = -1
        ys = 1
    elif xx > 0 and yy < 0:
        xs = 1
        ys = -1
    else:
        xs = 1
        ys = 1

    xx += width/2
    yy += height/2
    plt.plot([xx+2*xs*rr, xx+xs*rr], [yy, yy], color=contrast_col, lw=0.8)
    plt.plot([xx, xx], [yy+2*ys*rr, yy+ys*rr], color=contrast_col, lw=0.8)




# Constants, cgs
cc = 2.99792458e10 # cm s-1
GG = 6.67259e-8 # cm3 g-1 s-2
hh = 6.6260755e-27 # erg s
DD = 51.2 # kpc
pc = 3.086e18 # cm
kpc = 3.086e21 # cm
mpc = 3.086e24 # cm
kev2erg = 1.60218e-9 # erg keV-1
Msun = 1.989e33 # g
Lsun = 3.828e33 # erg s-1
Rsun = 6.957e10 # cm
Tsun = 5772 # K
uu = 1.660539040e-24 # g
SBc = 5.670367e-5 # erg cm-2 K-4 s-1
kB = 1.38064852e-16 # erg K-1
mp = 1.67262192369e-24 # g




            
################################################################
hips_width = 1200
hips_height = 1200
fov = 1/60.
contrast_col = '#66ff00'
#ff7900 https://en.wikipedia.org/wiki/Safety_orange

xrt = {'0149780101': 'XT 030206',
       '0203560201': 'XT 040610',
       '0300240501': 'XT 060207',
       '0300930301': 'XT 050925',
       '0502020101': 'XT 070618',
       '0604740101': 'XT 100424',
       '0675010401': 'XT 110621',
       '0743650701': 'XT 140811',
       '0760380201': 'XT 151128',
       '0765041301': 'XT 160220',
       '0770380401': 'XT 151219',
       '0781890401': 'XT 161028'}

host = {'XT 161028': [263.23707,  43.51231],
        'XT 151219': [173.53037,   0.87409],
        'XT 110621': [ 37.89582, -60.62918],
        'XT 030206': [ 29.28776,  37.62768],
        'XT 070618': None,
        'XT 060207': None,
        'XT 100424': [321.79659, -12.03900],
        'XT 151128': [167.07885,  -5.07495],
        'XT 050925': [311.43769, -67.64740],
        'XT 160220': [204.19926, -41.33718],
        'XT 140811': [ 43.65365,  41.07406],
        'XT 040610': None}
    
# Source (R, G, B)
objects = {'0149780101': ['CDS/P/PanSTARRS/DR1/color-z-zg-g'], # PanSTARRS
           '0203560201': ['CDS/P/PanSTARRS/DR1/color-z-zg-g'], # PanSTARRS (and DECaLS5)
           '0300240501': ['/Users/silver/dat/xmm/sbo/0300240501_sky/1_arcmin.png'], # VHS (Ks,J,J)
           # /Users/silver/dat/xmm/sbo/0300240501_sky/1_arcmin.png
           # CDS/P/2MASS/color
           
           '0300930301': ['/Users/silver/dat/xmm/sbo/0300930301_sky/1_arcmin.png'], # VHS (Ks,J,Y)
           # /Users/silver/dat/xmm/sbo/0300930301_sky/624_65_4_6453_{0:d}.fits
           # CDS/P/2MASS/color
           
           '0502020101': ['/Users/silver/dat/xmm/sbo/0502020101_sky/1_arcmin.png'], # Suprime (i', R, V)
           # /Users/silver/dat/xmm/sbo/0502020101_sky/1_arcmin.png
           # NOAO/P/DES/DR1/LIneA-color
           
           '0604740101': ['/Users/silver/dat/xmm/sbo/0604740101_sky/1_arcmin.png'], # Suprime (i', i', i')
           # /Users/silver/dat/xmm/sbo/0604740101_sky/1_arcmin.png
           # CDS/P/PanSTARRS/DR1/color-z-zg-g
           
           '0675010401': ['NOAO/P/DES/DR1/LIneA-color'], # DES
           '0743650701': ['CDS/P/PanSTARRS/DR1/color-z-zg-g'], # PanSTARRS
           '0760380201': ['CDS/P/PanSTARRS/DR1/color-z-zg-g'], # PanSTARRS
           '0765041301': ['/Users/silver/dat/xmm/sbo/0765041301_sky/1_arcmin.png'], # VHS (Ks,J,J)
           # /Users/silver/dat/xmm/sbo/0765041301_sky/026_391_7_31_6.fits
           # CDS/P/2MASS/color
           
           '0770380401': ['CDS/P/HSC/DR2/wide/color-i-r-g'], # HSC
           # CDS/P/HSC/DR2/wide/color-i-r-g
           # CDS/P/PanSTARRS/DR1/color-z-zg-g
           
           '0781890401': ['China-VO/P/BASS/DR3/image']} # BASS
           # China-VO/P/BASS/DR3/image
           # CDS/P/SDSS9/color
           # CDS/P/SDSS9/color-alt
           # CDS/P/PanSTARRS/DR1/color-z-zg-g

for oi in sorted(objects.keys()):
    ff = open('/Users/silver/box/phd/pro/sne/sbo/src/xmm_' + oi + '.sh')
    for line in ff:
        if line[:15] == '# uncertainty (':
            err = float(line.split()[-2].split('=')[-1])
        elif line[:9] == '# DET_ML=':
            ra = float(line.split(',')[1][2:])
            dec = float(line.split(',')[2][:-2])
    objects[oi].append(ra)
    objects[oi].append(dec)
    objects[oi].append(err)

for oi in sorted(objects.keys())[:]:
    ra = objects[oi][1]
    dec = objects[oi][2]

    if objects[oi][0][:7] == '/Users/':
        img = misc.imread(objects[oi][0])[:,:,:3]
        width, height, _ = img.shape
        if oi == '0502020101' or oi == '0604740101':
            img = img[::-1]
            src_db = 'Subaru/Suprime-Cam'
        else:
            src_db = 'VISTA(/VIRCAM) Hemisphere Survey'
        
        # rr = fits.open(objects[oi][0].format(1))
        # gg = fits.open(objects[oi][0].format(2))
        # bb = fits.open(objects[oi][0].format(3))
        # hdu = np.empty(rr[1].data.shape + (3,))
        # hdu[:,:,0] = rr[1].data
        # hdu[:,:,1] = gg[1].data
        # hdu[:,:,2] = bb[1].data
        # hdu = fits.PrimaryHDU(hdu)
        # width, height = rr[1].data.shape
    else:
        width = hips_width
        height = hips_height
        src_db = objects[oi][0]
        
        # HiPS
        url = 'http://alasky.u-strasbg.fr/hips-image-services/hips2fits?hips={}&width={}&height={}&fov={}&projection=TAN&coordsys=icrs&ra={}&dec={}'.format(quote(objects[oi][0]), width, height, fov, ra, dec)
        hdu = fits.open(url)
        url = 'http://alasky.u-strasbg.fr/hips-image-services/hips2fits?hips={}&width={}&height={}&fov={}&projection=TAN&coordsys=icrs&ra={}&dec={}'.format(quote(objects[oi][0]), width, height, 3*fov, ra, dec)
        out = fits.open(url)
        out[0].data = out[0].data.sum(axis=0).astype(np.float64)
        out[0].writeto('/Users/silver/box/phd/pro/sne/sbo/art/fig/' + oi + '_img.fits', overwrite=True)

        img = np.zeros((width, height, 3))
        img[:,:,0] = hdu[0].data[0]
        img[:,:,1] = hdu[0].data[1]
        img[:,:,2] = hdu[0].data[2]
        
        if '2MASS' in objects[oi][0]:
            contrast = 6.
        elif 'DES' in objects[oi][0]:
            contrast = 1.
        elif 'HSC' in objects[oi][0]:
            contrast = 1.2
            img = np.where(img < 20, 20, img)
            img = img - 20
        elif oi == '0743650701':
            contrast = 1.5
            img = np.where(img < 10, 10, img)
            img = img - 10
        elif 'PanSTARRS' in objects[oi][0]:
            contrast = 3.5
            img = np.where(img < 10, 10, img)
            img = img - 10
        else:
            contrast = 1.8
        
        img = contrast*img
        img = np.where(img > 255, 255, img)
        img = img/img.max()
        
    # Load XMM image
# xmm    con = fits.open(glob('/Users/silver/dat/xmm/sbo/' + oi + '_repro/' + oi + '_ep_?????_img_during.fits')[0])[0]
# xmm    wcs = get_wcs(con.header)
# xmm    pix = wcs.wcs_world2pix(ra,dec,0)
# xmm    cdelt = np.abs(con.header['CDELT1'])
# xmm    r1 = np.linspace(pix[1]-fov/cdelt/2, pix[1]+fov/cdelt/2, width)
# xmm    cdelt = np.abs(con.header['CDELT2'])
# xmm    r2 = np.linspace(pix[0]-fov/cdelt/2, pix[0]+fov/cdelt/2, height)
# xmm
# xmm    img = np.zeros((width, height))
# xmm    con = con.data
# xmm    for jj in range(0, width):
# xmm        for kk in range(0, height):
# xmm            img[jj,kk] = con[int(r1[jj]+0.5), int(r2[kk]+0.5)]
# xmm    con = gaussian_filter(img, height/fov*cdelt/2)
    
    # Plot
    file_name = '{}_img.fits'.format(oi)
    print('Saving {}'.format(file_name))
    fig = plt.figure(figsize=(2/3*5, 2/3*3.75))

    
    plt.imshow(img, origin='lower', interpolation='nearest')

# xmm    plt.contour(con, 8, colors='w', linewidths=0.5, alpha=0.5)
    if not host[xrt[oi]] is None:
        hra, hdec = SkyCoord(ra, dec, unit=u.deg).spherical_offsets_to(SkyCoord(*host[xrt[oi]], unit=u.deg))
        hra, hdec = hra.deg, hdec.deg
        hra = -hips_width/fov*hra # Minus because RA
        hdec = hips_height/fov*hdec
        mk_cross(hra, hdec)

    cir = plt.Circle((width/2, height/2), objects[oi][3]/3600./(fov/width), color=contrast_col, fill=False, zorder=100, lw=0.5)
    plt.gca().add_artist(cir)
    ax = plt.gca()
    ax.text(.5, .95, xrt[oi], horizontalalignment='center', transform=ax.transAxes, color=contrast_col)
    ax.text(.02, .02, src_db, transform=ax.transAxes, color=contrast_col)
    plt.axis('off')
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig('/Users/silver/box/phd/pro/sne/sbo/art/fig/' + oi + '_img.pdf', bbox_inches='tight', pad_inches=0.00001, dpi=100)
#    plt.show()
    plt.close()

db()
