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
from scipy.ndimage import gaussian_filter
import scipy.stats as sts
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy import units as u
from astropy.io import fits
from astropy import units

from astropy.cosmology import FlatLambdaCDM
cos = FlatLambdaCDM(H0=70, Om0=0.27, Tcmb0=2.725)

from sbo_mod import get_sbo_par

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

def get_zz(oi):
    ff = open('/Users/silver/dat/xmm/sbo/' + oi + '_repro/' + oi + '_fit_bb_during.log')
    for line in reversed(list(ff)):
        if line[12:31] == 'zTBabs     Redshift':
            return float(line.split()[4])

def get_mean2erg(oi):
    if gofs[oi + '_bb'] < gofs[oi + '_pl']:
        ff = open('/Users/silver/dat/xmm/sbo/' + oi + '_repro/' + oi + '_fit_bb_during.log')
    else:
        ff = open('/Users/silver/dat/xmm/sbo/' + oi + '_repro/' + oi + '_fit_pl_during.log')
    for line in reversed(list(ff)):
        if line[:12] == ' Model Flux ':
            return float(line.split()[4][1:])

def get_bb(oi):
    ff = open(oi + '_fit_bb_during.log')
    bfp = {}
    seen = False
    for line in reversed(list(ff)):
        if line[12:30] == 'clumin     lg10Lum':
            ll = float(line.split()[5])
            return tt, 10**ll, 10**bfp['bb_during_ll_low'], 10**bfp['bb_during_ll_upp']
        elif line[12:25] == 'zbbody     kT':
            tt = float(line.split()[5])

        elif line[:7] == '     7 ' and 'bb_during_ll_lim' in bfp.keys() and seen:
            bfp['bb_during_ll_low'] = float(line.split()[1])
            bfp['bb_during_ll_upp'] = float(line.split()[2])
        elif line[:7] == '     7 ':
            bfp['bb_during_ll_lim'] = float(line.split()[2])
            bfp['bb_during_ll_li2'] = float(line.split()[1])
            
        # elif line[:7] == '     8 ' and 'bb_during_tt_lim' in bfp.keys() and seen:
        #     bfp['bb_during_tt_low'] = float(line.split()[1])
        #     bfp['bb_during_tt_upp'] = float(line.split()[2])
        # elif line[:7] == '     8 ':
        #     bfp['bb_during_tt_lim'] = float(line.split()[2])
        #     bfp['bb_during_tt_li2'] = float(line.split()[1])
        elif line[:13] == 'XSPEC12>error':
            seen = True

        
################################################################
bin_par = {'0149780101': [15, 20],
           '0203560201': [200, 300],
           '0300240501': [3, 3],
           '0300930301': [40, 50],
           '0502020101': [2, 3],
           '0604740101': [100, 250],
           '0675010401': [12, 16],
           '0743650701': [100, 200],
           '0760380201': [40, 75],
           '0765041301': [40, 60],
           '0770380401': [6, 8],
           '0781890401': [2, 2]}
dur = {'0149780101': [160, 520],
           '0203560201': [800, 4000],
           '0300240501': [40, 150],
           '0300930301': [1300, 2150],
           '0502020101': [20, 90],
           '0604740101': [800, 6400],
           '0675010401': [190, 570],
           '0743650701': [1200, 3000],
           '0760380201': [750, 2550],
           '0765041301': [550, 1300],
           '0770380401': [330, 420],
           '0781890401': [28, 70]}

obs = ['0781890401',
       '0770380401',
       '0675010401',
       '0149780101',
       '0502020101',
       '0300240501',
       '0604740101',
       '0760380201',
       '0300930301',
       '0765041301',
       '0743650701',
       '0203560201']
xrt = {'0149780101': '030206',
       '0203560201': '040610',
       '0300240501': '060207',
       '0300930301': '050925',
       '0502020101': '070618',
       '0604740101': '100424',
       '0675010401': '110621',
       '0743650701': '140811',
       '0760380201': '151128',
       '0765041301': '160220',
       '0770380401': '151219',
       '0781890401': '161028'}

gofs = np.load('/Users/silver/box/phd/pro/sne/sbo/src/meta/gofs.npy',allow_pickle='TRUE').item()
   
buf  = '    \\colhead{XT} & \\colhead{$F_\\mathrm{mean}$}                   & \\colhead{$F_\\mathrm{peak}$}                   & \\colhead{$F_\\mathrm{before}$}                 & \\colhead{$F_\\mathrm{after}$}                  & \\colhead{Rise} & \\colhead{Decay}\\\\\n'
buf += '    \\colhead{}   & \\colhead{($10^{-13}$~erg~s$^{-1}$~cm$^{-2}$)} & \\colhead{($10^{-13}$~erg~s$^{-1}$~cm$^{-2}$)} & \\colhead{($10^{-13}$~erg~s$^{-1}$~cm$^{-2}$)} & \\colhead{($10^{-13}$~erg~s$^{-1}$~cm$^{-2}$)} & \\colhead{}     & \\colhead{}} \\startdata\n'
bu2  = '    \\colhead{XT} & \\colhead{$t$} & \\colhead{$L_\\mathrm{peak}$}        & \\colhead{$E_\\mathrm{SBO}$} & \\colhead{$R_{t}$} & \\colhead{$R_{E}$} &   \\colhead{$v_\\mathrm{sh}$} &   \\colhead{$v_\\mathrm{ej}$} &                 \\colhead{$\\rho$} & \\colhead{$E_\\mathrm{exp}$} \\\\\n'
bu2 += '    \\colhead{}   & \\colhead{(s)} & \\colhead{($10^{44}$~erg~s$^{-1}$)} & \\colhead{($10^{46}$~erg)}  & \\colhead{(\\Rsun)} & \\colhead{(\\Rsun)} & \\colhead{($10^{3}$\\kmps{})} & \\colhead{($10^{3}$\\kmps{})} & \\colhead{($10^{-9}$~g~cm$^{3}$)} & \\colhead{($10^{51}$~erg)}} \\startdata\n'
    
for oi in obs[:]:
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
    ff = open('/Users/silver/box/phd/pro/sne/sbo/src/xsp_mc_v2.sh')
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
    plt.figure(figsize=(2.95,3.25))
    pn = fits.open(oi + '_pn_py_lccorr.fits')[1].data
    has_m1 = True
    try:
        m1 = fits.open(oi + '_m1_py_lccorr.fits')[1].data
    except:
        has_m1 = False
    m2 = fits.open(oi + '_m2_py_lccorr.fits')[1].data

    tot = pn['RATE']+m1['RATE']+m2['RATE'] if has_m1 else pn['RATE']+m2['RATE']
    err = pn['ERROR']+m1['ERROR']+m2['ERROR'] if has_m1 else pn['ERROR']+m2['ERROR']
    tmp = pn['TIME']-t_rise
    sel = (tmp > 0) & (tmp < t_fade-t_rise)



    # Physical parameters
    vals = {}
    vals['tt'] = np.diff(dur[oi])[0]
    vals['zz'] = get_zz(oi)
    vals['mean2erg'] = get_mean2erg(oi)
    vals['TkeV'], vals['LL'], vals['Llol'], vals['Lupl'] = get_bb(oi)
    vals['EE'] = vals['LL']*vals['tt']/(1+vals['zz'])
    vals['TK'] = vals['TkeV']*kev2erg/kB
    vals['RBB'] =  np.sqrt(vals['LL']/(4*np.pi*SBc*vals['TK']**4))/Rsun
    vals['Rold'] = np.sqrt(vals['EE']/2.2e47)*1e13/Rsun 
    vals['REE'], vals['vbo'], vals['vej'], vals['rbo'], vals['Eexp'] = get_sbo_par(vals['EE'], vals['TkeV'])
    vals['REE'] = vals['REE']/Rsun
    vals['Rtt'] = cc*vals['tt']/(1+vals['zz'])/Rsun
    print('Energy:', vals['EE'], 'erg')
    print('Radius (BB):', vals['RBB'], 'Rsun')
    print('Radius (old):', vals['Rold'], 'Rsun')
    print('Radius (E):', vals['REE'], 'Rsun')
    print('Radius (t):', vals['Rtt'], 'Rsun')
    print('vbo: {0:.2e}, vej: {1:.2e}, rbo: {2:.2e}, Eexp: {3:.2e}'.format(vals['vbo'], vals['vej'], vals['rbo'], vals['Eexp']))
    vals['mean'] = [tot[sel].mean(), np.sqrt(np.sum(err[sel]**2))/sel.sum()]

    zhu = np.c_[tmp[sel], tot[sel], tot[sel]/vals['mean'][0]*vals['mean2erg']]
    np.savetxt('/Users/silver/box/sci/lib/a/alp20/zhu/' + oi + '_lc.txt', zhu)
    
    # 4 seconds
    # tmp = (tmp[1:]+tmp[:-1])/2.
    # tp = np.zeros(2*(tmp.size-1))
    # tp[::2] = tmp[:-1]
    # tp[1::2] = tmp[1:]
    # tot = tot[1:-1]
    # tmp = np.zeros(2*tot.size)
    # tmp[::2] = tot
    # tmp[1::2] = tot
    # plt.plot(tp, tmp, label='4 s', lw=1)



    # Larger bins
    tot = pn['RATE']+m1['RATE']+m2['RATE'] if has_m1 else pn['RATE']+m2['RATE']
    tmp = pn['TIME']-t_rise
    tmp = (tmp[1:]+tmp[:-1])/2.
    tmp = tmp[:tmp.size//bin_par[oi][0]*bin_par[oi][0]:bin_par[oi][0]]
    tp = np.zeros(2*(tmp.size-1))
    tp[::2] = tmp[:-1]
    tp[1::2] = tmp[1:]

    def bin_hlp(arr):
        arr = arr[1:-1]
        tmp = arr.size//bin_par[oi][0]
        arr = arr[:tmp*bin_par[oi][0]].reshape((tmp, bin_par[oi][0]))
        arr = arr.mean(1)
        tmp = np.zeros(2*arr.size)
        tmp[::2] = arr
        tmp[1::2] = arr
        return tmp
    tmp = bin_hlp(tot)
    def bin_hlp_err(arr):
        arr = arr[1:-1]
        tmp = arr.size//bin_par[oi][0]
        arr = arr[:tmp*bin_par[oi][0]].reshape((tmp, bin_par[oi][0]))
        arr = np.sqrt(np.sum(arr**2, axis=1))/bin_par[oi][0]
        return arr
    err = bin_hlp_err(err)
    # print(tp.shape, tmp.shape)
    tmp = tmp[:tp.size]
    plt.plot(tp, tmp, label=str(bin_par[oi][0]*4) + ' s', lw=1)
    plt.errorbar((tp[::2]+tp[1::2])/2., tmp[::2], yerr=err[:tmp[::2].size], fmt=None, color='#6aafe7', lw=1)
    
    # Smoothed
    tot = pn['RATE']+m1['RATE']+m2['RATE'] if has_m1 else pn['RATE']+m2['RATE']
    tmp = pn['TIME']-t_rise
    plt.plot(tmp, gaussian_filter(tot, bin_par[oi][1]), label='$\sigma=' + str(bin_par[oi][1]*4) + '$ s', lw=1)
    
    print('Average rate:', tot[(pn['TIME'] > t_rise) & (pn['TIME'] < t_fade)].mean(), 'cts s-1')

    # Really nasty hack to cheat into legend
# HR    plt.plot(0, 0, color='gray', ls='--', label='HR', zorder=-1000)

    # Peak
    pnp = fits.open(oi + '_pn_lccorr_peak.fits')[1].data
    if has_m1: m1p = fits.open(oi + '_m1_lccorr_peak.fits')[1].data
    m2p = fits.open(oi + '_m2_lccorr_peak.fits')[1].data
    rp = float(pnp['RATE']+m1p['RATE']+m2p['RATE']) if has_m1 else float(pnp['RATE']+m2p['RATE'])
    rt = (tpeak_min+tpeak_max)/2.*np.ones(2)-t_rise
    ep = float(np.sqrt(pnp['ERROR']**2+m1p['ERROR']**2+m2p['ERROR']**2)) if has_m1 else float(np.sqrt(pnp['ERROR']**2+m2p['ERROR']**2))
    plt.plot(np.array([tpeak_min, tpeak_max])-t_rise, [rp, rp], 'k')
    plt.plot(rt, [rp-ep, rp+ep], 'k')
    print('Peak: {0:.6f}+-{1:.6f}'.format(rp, ep))

    vals['peak'] = [rp, ep]
    
    # Limits/before+after
    if bsep > 10:
        plt.plot(np.array([0, 0.1*(t_fade-t_rise)]), [b_lim, b_lim], 'k')
        # plt.annotate('$<{0:.3f}$'.format(b_lim), (0.1*rt[0], 0.05*rp))
        print('Dynamic range rise: >{0:.2f}'.format(rp/b_lim))
        vals['before'] = [b_lim]
    else:
        plt.plot(np.array([0, 0.1*(t_fade-t_rise)]), [brr, brr], 'k')
        plt.plot(np.ones(2)*0.05*(t_fade-t_rise), [brr-bee, brr+bee], 'k')
        # plt.annotate('${0:.3f}\pm{1:.3f}$'.format(brr, bee), (0.1*rt[0], 0.05*rp))
        print('Dynamic range rise: {0:.2f}'.format(rp/brr))
        vals['before'] = [brr, bee]
    
    if asep > 10:
        plt.plot(np.array([t_fade, t_fade-0.1*(t_fade-t_rise)])-t_rise, [a_lim, a_lim], 'k')
        # plt.annotate('$<{0:.3f}$'.format(a_lim), ((t_fade-t_rise)*0.75, 0.05*rp))
        print('Dynamic range fade: >{0:.2f}'.format(rp/a_lim))
        vals['after'] = [a_lim]
    else:
        plt.plot(np.array([t_fade, t_fade-0.1*(t_fade-t_rise)])-t_rise, [arr, arr], 'k')
        plt.plot(np.ones(2)*(t_fade-t_rise-0.05*(t_fade-t_rise)), [arr-aee, arr+aee], 'k')
        # plt.annotate('${0:.3f}\pm{1:.3f}$'.format(arr, aee), ((t_fade-t_rise)*0.75, 0.05*rp))
        print('Dynamic range fade: {0:.2f}'.format(rp/arr))
        vals['after'] = [arr, aee]

# HR    # Hardness
# HR    t_hard = np.linspace(t_rise, t_fade, 7)
# HR    soft, _ = np.histogram(tt[pi < 1.], t_hard)
# HR    hard, _ = np.histogram(tt[pi >= 1.], t_hard)
# HR    # https://xmm-tools.cosmos.esa.int/external/sas/current/doc/emldetect/node3.html
# HR    hr = np.where(soft+hard > 0., (hard-soft)/(hard+soft), 0.)
# HR    var = np.empty(hr.size)
# HR    nn = 10000
# HR    for ii in range(0, var.size):
# HR        if soft[ii]+hard[ii] > 0.:
# HR            aa = sts.poisson(hard[ii]).rvs(nn)
# HR            bb = sts.poisson(soft[ii]).rvs(nn)
# HR            var[ii] = np.std(np.where(aa+bb==0, 0., (aa-bb)/(aa+bb)))
# HR        else:
# HR            print('\n\n\n\nWARNING: EMPTY HARDNESS BIN\n\n\n')
# HR            var[ii] = 0.
# HR    print(var)
# HR    ax1 = plt.gca()
# HR    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
# HR    ax2.set_ylabel('Hardness')  # we already handled the x-label with ax1
# HR    eb2 = ax2.errorbar((t_hard[1:]+t_hard[:-1])/2.-t_rise, hr, yerr=var, color='gray', fmt=None)
# HR    eb2[-1][0].set_linestyle('--')
# HR
# HR    def bin_hlp(arr):
# HR        tmp = np.empty(2*arr.size-2)
# HR        tmp[::2] = arr[:-1]
# HR        tmp[1::2] = arr[1:]
# HR        return tmp
# HR    t_hard = bin_hlp(t_hard)-t_rise
# HR
# HR    def bin_hlp(arr):
# HR        tmp = np.empty(2*arr.size)
# HR        tmp[::2] = arr
# HR        tmp[1::2] = arr
# HR        return tmp
# HR    ax2.plot(t_hard, bin_hlp(hr), color='gray', ls='--')
# HR    ax2.set_ylim([-1, 1])
# HR        
# HR    # Cosmetics
# HR    plt.sca(ax1)
    plt.legend() 
    plt.xlabel('$t$ (s)')
    plt.ylabel('Rate (counts s$^{-1}$)')
    # plt.annotate('${0:.3f}\pm{1:.3f}$'.format(rp, ep), (1.1*rt[0], 1.1*rp))
    # plt.gca().axvspan(t0-t_rise, t1-t_rise, alpha=0.4, color='#8c564b', ec=None)
    plt.gca().axvspan(dur[oi][0], dur[oi][1], alpha=0.2, color='#7f7f7f', ec=None, zorder=-9999)
    plt.gca().axvline(t_mid-t_rise, ls='--', color='#2ca02c', zorder=-9000)
    # plt.gca().axhline(vals['mean'][0], ls='--', color='#d62728')
    plt.xlim([0, t_fade-t_rise])
    if t_fade-t_rise == 1000 or t_fade-t_rise == 10000 or t_fade-t_rise == 5000 or t_fade-t_rise == 6000:
        plt.xticks(plt.xticks()[0][:-1], plt.xticks()[0][:-1].astype('int'))

    # plt.ylim([-0.1*(rp+ep), 1.1*(rp+ep)])
    plt.ylim([0., 1.1*(rp+ep)])
    if oi == '0502020101': plt.yticks(plt.yticks()[0], ['{0:.1f}'.format(y) for y in plt.yticks()[0]])
    else: plt.yticks(plt.yticks()[0], ['{0:.2f}'.format(y) for y in plt.yticks()[0]])
    np.savetxt('/Users/silver/box/phd/pro/sne/sbo/art/fig/' + oi + '_ep_lccorr.txt', np.c_[pn['TIME'], pn['RATE']+m1['RATE']+m2['RATE']]) if has_m1 else np.savetxt('/Users/silver/box/phd/pro/sne/sbo/art/fig/' + oi + '_ep_lccorr.txt', np.c_[pn['TIME'], pn['RATE']+m2['RATE']])
    plt.savefig('/Users/silver/box/phd/pro/sne/sbo/art/fig/' + oi + '_ep_lccorr.pdf', bbox_inches='tight', pad_inches=0.1, dpi=300)
    # plt.show()
    # db()
    plt.close()


    

    # Scatter plot
    # plt.figure(figsize=(5,3.75))
    # plt.semilogy(tt-t_rise, pi, '.k')
    # plt.title(oi)
    # plt.xlabel('$t-t_{rise}$ (s)')
    # plt.ylabel('Energy (keV)')
    # plt.ylim([0.1, 10.])
    # # plt.gca().axvspan(t0-t_rise, t1-t_rise, alpha=0.2, color='gray', ec=None)
    # plt.gca().axvspan(dur[oi][0], dur[oi][1], alpha=0.2, color='#7f7f7f', ec=None)
    # np.savetxt('/Users/silver/box/phd/pro/sne/sbo/art/fig/' + oi + '_scatter.txt', np.c_[tt-t_rise, pi])
    # plt.savefig('/Users/silver/box/phd/pro/sne/sbo/art/fig/' + oi + '_scatter.pdf', bbox_inches='tight', pad_inches=0.1, dpi=300)
    # # plt.show()
    # plt.close()
    # # db()



    def fmt_val(lab):
        ce = vals['mean2erg']/vals['mean'][0]*1.e13
        if lab == 'rise':
            tmp = vals['peak'][0]/vals['before'][0]
            if len(vals['before']) == 1: return '$>{0:.0f}$'.format(tmp)
            else: return'${0:.0f}$'.format(tmp)
        elif lab == 'decay':
            tmp = vals['peak'][0]/vals['after'][0]
            if len(vals['after']) == 1: return '$>{0:.0f}$'.format(tmp)
            else: return'${0:.0f}$'.format(tmp)
        elif lab == 'mean' or lab == 'peak' or ((lab == 'before' or lab == 'after') and len(vals[lab]) == 2):
            return '${0:.2f}\pm{1:.2f}$'.format(ce*vals[lab][0], ce*vals[lab][1])
        else:
            return '$<{0:.2f}$'.format(ce*vals[lab][0])

    buf += '  {0:15s}& {1:46s}& {2:46s}& {3:46s}& {4:46s}& {5:15s}& {6:15s}\\\\\n'.format(
        xrt[oi], fmt_val('mean'), fmt_val('peak'), fmt_val('before'),
        fmt_val('after'), fmt_val('rise'), fmt_val('decay'))

    def fmt_va2(lab):

        scale = vals['peak'][0]/vals['mean'][0]/1e44
        if lab == 'LL':
            aa = scale*vals['LL']
            bb = scale*vals['Llol']
            cc = scale*vals['Lupl']
            return '${0:.2f}_{{{1:.2f}}}^{{+{2:.2f}}}$'.format(aa, bb-aa, cc-aa)
        elif lab == 'EE':
            aa = vals['EE']/1.e46
            bb = vals['Llol']*vals['tt']/(1+vals['zz'])/1.e46
            cc = vals['Lupl']*vals['tt']/(1+vals['zz'])/1.e46
            return '${0:.1f}_{{{1:.1f}}}^{{+{2:.1f}}}$'.format(aa, bb-aa, cc-aa)
            
    bu2 += '  {0:15s}& {1:13d} & {2:34s} & {3:26s} & {4:17d} & {5:17d} & {6:27.0f} & {7:27.1f} & {8:32.1f} & {9:26.1f} \\\\\n'.format(
        xrt[oi],
        vals['tt'],
        fmt_va2('LL'),
        fmt_va2('EE'),
        int(np.round(vals['Rtt'])),
        int(np.round(vals['REE'])),
        vals['vbo']/1.e8,
        vals['vej']/1.e8,
        vals['rbo']/1.e-9,
        vals['Eexp']/1.e51)

print('\n\n\n\n\n')
print(buf)
print(bu2)
# db()
