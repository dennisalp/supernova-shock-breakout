'''
2020-01-10, Dennis Alp, dalp@kth.se

Extract all light curve properties.
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
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy import units as u
from astropy.io import fits
from astropy import units

#For LaTeX style font in plots
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rc('text', usetex=True)

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

# print(dat.dtype.names)

################################################################
obs = ['0149780101',
       '0203560201',
       '0300240501',
       '0300930301',
       '0502020101',
       '0604740101',
       '0675010401',
       '0743650701',
       '0760380201',
       '0765041301',
       '0770380401',
       '0781890401']

# '0770380201.sh'
# '0802860201.sh'


    
for oi in obs:
    # Get coordinates
    print('\n\n' + oi)
    ff = open('/Users/silver/box/phd/pro/sne/sbo/src/xmm_' + oi + '.sh')
    for line in ff:
        if line[:7] == 't_rise=':
            t_rise = float(line.split('=')[1])
        elif line[:6] == 't_mid=':
            t_mid = float(line.split('=')[1])
        elif line[:7] == 't_fade=':
            t_fade = float(line.split('=')[1])
        elif line[:9] == 'tpeak_min':
            tpeak_min = float(line.split('=')[1])
        elif line[:9] == 'tpeak_max':
            tpeak_max = float(line.split('=')[1])
        elif line[:9] == 'tpeak_bin':
            tpeak_bin = float(line.split('=')[1])
        elif line[:9] == '# DET_ML=':
            ra = float(line.split(',')[1][2:])
            de = float(line.split(',')[2][:-2])
    coo = SkyCoord(ra*u.deg, de*u.deg)
    print('Total duration:', t_fade-t_rise, 's')

    # Get N_H
    ff = open('/Users/silver/box/phd/pro/sne/sbo/src/xsp_mc.sh')
    trigger = False
    for line in ff:
        if line[:13] == 'oi=' + oi:
            trigger = True
        elif trigger:
            nh = float(line.strip().split('=')[1])
            break
    print('N_H:', nh*1.e22, 'cm^-2')

    
    # Load the data
    os.chdir('/Users/silver/dat/xmm/sbo/' + oi + '_repro/')
    b_ls = fits.open(oi + '_emllist_before.fits')
    d_ls = fits.open(oi + '_emllist_during.fits')
    a_ls = fits.open(oi + '_emllist_after.fits')
    b_lim = fits.open(oi + '_ep_clean_img_beforesen.fits')
    try:
        d_lim = fits.open(oi + '_ep_dirty_img_duringsen.fits')
    except:
        d_lim = fits.open(oi + '_ep_clean_img_duringsen.fits')
    a_lim = fits.open(oi + '_ep_clean_img_aftersen.fits')


    
    # Get limits
    def get_lim(fz):
        wcs = WCS()
        wcs.wcs.crpix = [fz[0].header['CRPIX1'], fz[0].header['CRPIX2']]
        wcs.wcs.cdelt = [fz[0].header['CDELT1'], fz[0].header['CDELT2']]
        wcs.wcs.crval = [fz[0].header['CRVAL1'], fz[0].header['CRVAL2']]
        wcs.wcs.ctype = [fz[0].header['CTYPE1'], fz[0].header['CTYPE2']]
        x, y = wcs.wcs_world2pix(ra, de, 1)
        x = int(np.round(x))
        y = int(np.round(y))
        return fz[0].data[y, x]

    b_lim = get_lim(b_lim)
    d_lim = get_lim(d_lim)
    a_lim = get_lim(a_lim)
    print(b_lim, d_lim, a_lim)



    
    # Get detections
    def get_det(coo, fz):
        det = SkyCoord(fz[1].data['RA']*u.deg, fz[1].data['DEC']*u.deg)
        keep = fz[1].data['ID_INST']==0
        det = det[keep]
        sep = coo.separation(det)
        sepi = sep.argmin()
        sep = sep[sepi].arcsec
        rr = fz[1].data['RATE'][keep][sepi]
        ee = fz[1].data['RATE_ERR'][keep][sepi]
        return rr, ee, sep

    brr, bee, bsep = get_det(coo, b_ls)
    drr, dee, dsep = get_det(coo, d_ls)
    arr, aee, asep = get_det(coo, a_ls)
    print('{0:.6f}+-{1:.6f} at {2:9.3f} arcsec'.format(brr, bee, bsep))
    print('{0:.6f}+-{1:.6f} at {2:9.3f} arcsec'.format(drr, dee, dsep))
    print('{0:.6f}+-{1:.6f} at {2:9.3f} arcsec'.format(arr, aee, asep))


    

    ################################################################
    # Duration
    ff = glob('{0:s}_ep_*_evt_src_during.fits'.format(oi))
    if not len(ff) == 1:
        print(ff)
        gg=wp
    ff = ff[0]
    dat = fits.open(ff)[1].data
    tt = dat['TIME']
    pi = dat['PI']/1.e3
    


    ii = tt.argsort()
    tt = tt[ii]
    tdur = 70
    cts = np.round(tdur/100.*tt.size).astype('int')
    durations = tt[cts:]-tt[:-cts]
    ii = np.argmin(durations)
    t0 = tt[ii]
    t1 = tt[ii+cts]
    print('T{4:d}: {0:d} counts within {1:f} seconds between {2:f} and {3:f}'.format(cts, t1-t0, t0, t1, tdur))




    ################################################################
    # Plot
    plt.figure(figsize=(2.75,3.75))
    pn = fits.open(oi + '_pn_lccorr.fits')[1].data
    plt.plot(pn['TIME']-t_rise, pn['RATE'], label='pn')
    has_m1 = True
    try:
        m1 = fits.open(oi + '_m1_lccorr.fits')[1].data
        plt.plot(m1['TIME']-t_rise, m1['RATE'], label='M1')
    except:
        has_m1 = False
        plt.plot([], [])
    m2 = fits.open(oi + '_m2_lccorr.fits')[1].data
    plt.plot(m2['TIME']-t_rise, m2['RATE'], label='M2')
    tot = pn['RATE']+m1['RATE']+m2['RATE'] if has_m1 else pn['RATE']+m2['RATE']
    plt.plot(pn['TIME']-t_rise, tot, label='Sum')
    print('Average rate:', tot[(pn['TIME'] > t_rise) & (pn['TIME'] < t_fade)].mean(), 'cts s-1')
    # Really nasty hack to cheat into legend
    plt.plot(0, 0, color='gray', ls='--', label='HR')
    
    pnp = fits.open(oi + '_pn_lccorr_peak.fits')[1].data
    if has_m1: m1p = fits.open(oi + '_m1_lccorr_peak.fits')[1].data
    m2p = fits.open(oi + '_m2_lccorr_peak.fits')[1].data
    rp = float(pnp['RATE']+m1p['RATE']+m2p['RATE']) if has_m1 else float(pnp['RATE']+m2p['RATE'])
    rt = (tpeak_min+tpeak_max)/2.*np.ones(2)-t_rise
    ep = float(np.sqrt(pnp['ERROR']**2+m1p['ERROR']**2+m2p['ERROR']**2)) if has_m1 else float(np.sqrt(pnp['ERROR']**2+m2p['ERROR']**2))
    plt.plot(np.array([tpeak_min, tpeak_max])-t_rise, [rp, rp], 'k')
    plt.plot(rt, [rp-ep, rp+ep], 'k')
    print('Peak: {0:.6f}+-{1:.6f}'.format(rp, ep))

    # Limits/before+after
    if bsep > 10:
        plt.plot(np.array([0, 0.1*(t_fade-t_rise)]), [b_lim, b_lim], 'k')
        plt.annotate('$<{0:.3f}$'.format(b_lim), (0.1*rt[0], -0.1*rp))
        print('Dynamic range rise: >{0:.2f}'.format(rp/b_lim))
    else:
        plt.plot(np.array([0, 0.1*(t_fade-t_rise)]), [brr, brr], 'k')
        plt.plot(np.ones(2)*0.05*(t_fade-t_rise), [brr-bee, brr+bee], 'k')
        plt.annotate('${0:.3f}\pm{1:.3f}$'.format(brr, bee), (0.1*rt[0], -0.1*rp))
        print('Dynamic range rise: {0:.2f}'.format(rp/brr))
    
    if asep > 10:
        plt.plot(np.array([t_fade, t_fade-0.1*(t_fade-t_rise)])-t_rise, [a_lim, a_lim], 'k')
        plt.annotate('$<{0:.3f}$'.format(a_lim), ((t_fade-t_rise)*0.85, -0.1*rp))
        print('Dynamic range fade: >{0:.2f}'.format(rp/a_lim))
    else:
        plt.plot(np.array([t_fade, t_fade-0.1*(t_fade-t_rise)])-t_rise, [arr, arr], 'k')
        plt.plot(np.ones(2)*(t_fade-t_rise-0.05*(t_fade-t_rise)), [arr-aee, arr+aee], 'k')
        plt.annotate('${0:.3f}\pm{1:.3f}$'.format(arr, aee), ((t_fade-t_rise)*0.8, -0.1*rp))
        print('Dynamic range fade: {0:.2f}'.format(rp/arr))

    # Hardness
    t_hard = np.linspace(t_rise, t_fade, 5)
    soft, _ = np.histogram(tt[pi < 1.], t_hard)
    hard, _ = np.histogram(tt[pi >= 1.], t_hard)
    # https://xmm-tools.cosmos.esa.int/external/sas/current/doc/emldetect/node3.html
    hr = np.where(soft+hard > 0., (hard-soft)/(hard+soft), 0.)
    ax1 = plt.gca()
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('Hardness')  # we already handled the x-label with ax1
    ax2.plot((t_hard[1:]+t_hard[:-1])/2.-t_rise, hr, color='gray', ls='--')
    ax2.set_ylim([-1, 1])
        
    # Cosmetics
    plt.sca(ax1)
    plt.legend() 
#    plt.title(oi)
    plt.xlabel('$t-t_{rise}$ (s)')
    plt.ylabel('Rate (cts/s)')
    plt.annotate('${0:.3f}\pm{1:.3f}$'.format(rp, ep), (1.1*rt[0], 1.1*rp))
    plt.gca().axvspan(t0-t_rise, t1-t_rise, alpha=0.2, color='gray')
    plt.xlim([0, t_fade-t_rise])
    plt.ylim([-0.1*(rp+ep), 1.1*(rp+ep)])
    np.savetxt('/Users/silver/box/phd/pro/sne/sbo/art/fig/' + oi + '_ep_lccorr.txt', np.c_[pn['TIME'], pn['RATE']+m1['RATE']+m2['RATE']]) if has_m1 else np.savetxt('/Users/silver/box/phd/pro/sne/sbo/art/fig/' + oi + '_ep_lccorr.txt', np.c_[pn['TIME'], pn['RATE']+m2['RATE']])
    plt.savefig('/Users/silver/box/phd/pro/sne/sbo/art/fig/' + oi + '_ep_lccorr.pdf', bbox_inches='tight', pad_inches=0.1, dpi=300)
    # plt.show()
    # db()
    plt.close()



    

    # Scatter plot
    plt.figure(figsize=(5,3.75))
    plt.semilogy(tt-t_rise, pi, '.k')
    plt.title(oi)
    plt.xlabel('$t-t_{rise}$ (s)')
    plt.ylabel('Energy (keV)')
    plt.ylim([0.1, 10.])
    plt.gca().axvspan(t0-t_rise, t1-t_rise, alpha=0.2, color='gray')
    np.savetxt('/Users/silver/box/phd/pro/sne/sbo/art/fig/' + oi + '_scatter.txt', np.c_[tt-t_rise, pi])
    plt.savefig('/Users/silver/box/phd/pro/sne/sbo/art/fig/' + oi + '_scatter.pdf', bbox_inches='tight', pad_inches=0.1, dpi=300)
    # plt.show()
    plt.close()
    # db()
    
# db()
