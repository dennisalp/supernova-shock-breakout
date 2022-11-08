'''
2020-01-13, Dennis Alp, dalp@kth.se

Compute the parameter distributions from the XSPEC MC output.
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
import scipy.stats as sts
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy import units as u
from astropy.io import fits
from astropy import units

#For LaTeX style font in plots
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rc('text', usetex=True)





################################################################
# Help functions
def get_bf():
    bfp = {}

    ff = open(oi + '_fit_bb_during.log')
    for line in reversed(list(ff)):
        if line[12:25] == 'zTBabs     nH':
            bfp['bb_during_nh'] = float(line.split()[5])
            break
        elif line[12:31] == 'zTBabs     Redshift':
            bfp['bb_during_zz'] = float(line.split()[4])
        elif line[12:25] == 'zbbody     kT':
            bfp['bb_during_tt'] = float(line.split()[5])
        elif line[:40] == 'Test statistic : log(Cramer-von Mises) =':
            bfp['bb_during_gof'] = float(line.split()[6])

    ff = open(oi + '_fit_bb_first.log')
    for line in reversed(list(ff)):
        if line[12:25] == 'zbbody     kT':
            bfp['bb_first_tt'] = float(line.split()[5])
            break
        elif line[:40] == 'Test statistic : log(Cramer-von Mises) =':
            bfp['bb_first_gof'] = float(line.split()[6])
            
    ff = open(oi + '_fit_bb_second.log')
    for line in reversed(list(ff)):
        if line[12:25] == 'zbbody     kT':
            bfp['bb_second_tt'] = float(line.split()[5])
            break
        elif line[:40] == 'Test statistic : log(Cramer-von Mises) =':
            bfp['bb_second_gof'] = float(line.split()[6])

    # Power law
    ff = open(oi + '_fit_pl_during.log')
    for line in reversed(list(ff)):
        if line[12:25] == 'zTBabs     nH':
            bfp['pl_during_nh'] = float(line.split()[5])
            break
        elif line[12:31] == 'zpowerlw   PhoIndex':
            bfp['pl_during_gg'] = float(line.split()[4])
        elif line[12:27] == 'zpowerlw   norm':
            bfp['pl_during_tt'] = float(line.split()[4])
        elif line[:40] == 'Test statistic : log(Cramer-von Mises) =':
            bfp['pl_during_gof'] = float(line.split()[6])
        
    ff = open(oi + '_fit_pl_first.log')
    for line in reversed(list(ff)):
        if line[12:31] == 'zpowerlw   PhoIndex':
            bfp['pl_first_gg'] = float(line.split()[4])
            break
        elif line[:40] == 'Test statistic : log(Cramer-von Mises) =':
            bfp['pl_first_gof'] = float(line.split()[6])
        
    ff = open(oi + '_fit_pl_second.log')
    for line in reversed(list(ff)):
        if line[12:31] == 'zpowerlw   PhoIndex':
            bfp['pl_second_gg'] = float(line.split()[4])
            break
        elif line[:40] == 'Test statistic : log(Cramer-von Mises) =':
            bfp['pl_second_gof'] = float(line.split()[6])

    return bfp
        

def parse_bb_log(fp):
    ff = open(fp)
    # Parse each log
    for line in reversed(list(ff)):
        if line[12:25] == 'zTBabs     nH':
            nh = float(line.split()[5])
            return nh, zz, tt, gof
            # try: # THESE ARE TEMPORARY REMOVE NOW I FIEXED THE ILL POSED PROBEM IN THE XSPEC MC
            #     return nh, zz, tt, gof
            # except:
            #     return nh, zz, tt, -5.
        elif line[12:31] == 'zTBabs     Redshift':
            zz = float(line.split()[4])
        elif line[12:25] == 'zbbody     kT':
            tt = float(line.split()[5])
        elif line[:40] == 'Test statistic : log(Cramer-von Mises) =':
            gof = float(line.split()[6])
            

def parse_pl_log(fp):
    ff = open(fp)
    # Parse each log
    for line in reversed(list(ff)):
        if line[12:25] == 'zTBabs     nH':
            nh = float(line.split()[5])
            return nh, gg, kk, gof
            # try:
            #     return nh, gg, kk, gof
            # except:
            #     return nh, gg, kk, -5.
        elif line[12:31] == 'zpowerlw   PhoIndex':
            gg = float(line.split()[4])
        elif line[12:27] == 'zpowerlw   norm':
            kk = float(line.split()[4])
        elif line[:40] == 'Test statistic : log(Cramer-von Mises) =':
            gof = float(line.split()[6])

def get_lim(val, dist):
    rank = sts.percentileofscore(dist, val)

    if rank < 100.*(sts.norm.cdf(1)-.5): # Upper limit
        low = np.nan
        up = np.percentile(dist, 100.-100.*2*sts.norm.sf(3))
        
    elif rank > 100.-100.*(sts.norm.cdf(1)-.5): # Lower limit
        low = np.percentile(dist, 100.*2*sts.norm.cdf(-3))
        up = np.nan

    else:
        low = np.percentile(dist, rank-100.*(sts.norm.sf(-1)-.5))
        up = np.percentile(dist, rank+100.*(sts.norm.cdf(1)-.5))

    return low, up
    
def fmt_val(lab):
    val = bfp[lab]
    low = bfp[lab+'_low']
    up = bfp[lab+'_up']
    if np.isnan(low):
        return '$<{0:.2f}$'.format(up)
    elif np.isnan(up):
        return '$>{0:.2f}$'.format(low)
    else:
        return '${0:.2f}_{{{1:.2f}}}^{{+{2:.2f}}}$'.format(val, low-val, up-val)




################################################################
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

obs = ['0149780101',
       '0203560201',
       '0300240501',
       '0300930301',
       '0502020101',
       '0555780101',
       '0604740101',
       '0651690101',
       '0675010401',
       '0743650701',
       '0760380201',
       '0765041301',
       '0770380401',
       '0781890401']

# '0770380201.sh'
# '0802860201.sh'
    




################################################################
# Parse the MC logs
print('\\colhead{Obs. ID}                  & \\colhead{$z_\\mathrm{f}$}           & \\colhead{$N_\\mathrm{H,bb}$}        & \\colhead{$T$}                      & \\colhead{$T_1$}                    & \\colhead{$T_2$}                    & \\colhead{$N_\\mathrm{H,pl}$}        & \\colhead{$\\Gamma$}                 & \\colhead{$\\Gamma_1$}               & \\colhead{$\\Gamma_2$}               \\\\')
print('\\colhead{}                         & \\colhead{}                         & \\colhead{($10^{22}$~cm$^{-2}$)}    & \\colhead{(keV)}                    & \\colhead{(keV)}                    & \\colhead{(keV)}                    & \\colhead{($10^{22}$~cm$^{-2}$)}    & \\colhead{}                         & \\colhead{}                         & \\colhead{}                         \\\\')
for oi in obs[:]:

    os.chdir('/Users/silver/dat/xmm/sbo/' + oi + '_repro/')
    bfp = get_bf()
    
    # Blackbody
    os.chdir('/Users/silver/dat/xmm/sbo/' + oi + '_repro/bb_log/')
    files = sorted(glob('*.log'))
    bbp = np.zeros((len(files)//3, 3*4))

    # All logs
    for ii, ff in enumerate(files):
        row = np.mod(ii, len(files)//3)
        col = ii//(len(files)//3)
        bbp[row, 4*col:4*col+4] = parse_bb_log(ff)
    np.savetxt('bb_par_during.txt', bbp[:,:4])
    np.savetxt('bb_par_first.txt', bbp[:,4:8])
    np.savetxt('bb_par_second.txt', bbp[:,8:])

    # Start with power law
    os.chdir('/Users/silver/dat/xmm/sbo/' + oi + '_repro/pl_log/')
    files = sorted(glob('*.log'))
    plp = np.zeros((len(files)//3, 3*4))

    # All logs
    for ii, ff in enumerate(files):
        row = np.mod(ii, len(files)//3)
        col = ii//(len(files)//3)
        plp[row, 4*col:4*col+4] = parse_pl_log(ff)
    np.savetxt('pl_par_during.txt', plp[:,:4])
    np.savetxt('pl_par_first.txt', plp[:,4:8])
    np.savetxt('pl_par_second.txt', plp[:,8:])

    

    ################################################################
    # Table
    bfp['bb_during_zz_low'], bfp['bb_during_zz_up'] = get_lim(bfp['bb_during_zz'], bbp[:,1])
    bfp['bb_during_nh_low'], bfp['bb_during_nh_up'] = get_lim(bfp['bb_during_nh'], bbp[:,0])
    bfp['bb_during_tt_low'], bfp['bb_during_tt_up'] = get_lim(bfp['bb_during_tt'], bbp[:,2])
    bfp['bb_first_tt_low'], bfp['bb_first_tt_up'] = get_lim(bfp['bb_first_tt'], bbp[:,6])
    bfp['bb_second_tt_low'], bfp['bb_second_tt_up'] = get_lim(bfp['bb_second_tt'], bbp[:,10])
    
    bfp['pl_during_nh_low'], bfp['pl_during_nh_up'] = get_lim(bfp['pl_during_nh'], plp[:,0])
    bfp['pl_during_gg_low'], bfp['pl_during_gg_up'] = get_lim(bfp['pl_during_gg'], plp[:,1])
    bfp['pl_first_gg_low'], bfp['pl_first_gg_up'] = get_lim(bfp['pl_first_gg'], plp[:,5])
    bfp['pl_second_gg_low'], bfp['pl_second_gg_up'] = get_lim(bfp['pl_second_gg'], plp[:,9])

    print('{9:35s}& {0:35s}& {1:35s}& {2:35s}& {3:35s}& {4:35s}& {5:35s}& {6:35s}& {7:35s}& {8:35s}\\\\'.format(fmt_val('bb_during_zz'), fmt_val('bb_during_nh'), fmt_val('bb_during_tt'), fmt_val('bb_first_tt'), fmt_val('bb_second_tt'), fmt_val('pl_during_nh'), fmt_val('pl_during_gg'), fmt_val('pl_first_gg'), fmt_val('pl_second_gg'), oi))
    
    ################################################################
    # Plots
    os.chdir('/Users/silver/dat/xmm/sbo/' + oi + '_repro/')
    ng = np.linspace(0, 25, 100)
    zg = np.logspace(-3, 1, 100)
    lg = np.linspace(38, 48, 100)
    gg = np.linspace(-1, 8, 100)
    gofg = np.linspace(-20, 10, 100)
    tg = np.logspace(np.log10(0.03), np.log10(3), 100)
    
    plt.figure(figsize=(5,3.75))
    plt.hist(np.log10(bbp[:,0]*1.e22), ng, normed=True, alpha=0.3)
    plt.hist(np.log10(plp[:,0]*1.e22), ng, normed=True, alpha=0.3)
    plt.title(oi)
    plt.legend(['bb', 'pl'])
    plt.gca().axvline(np.log10(bfp['bb_during_nh']*1.e22), color='#1f77b4')
    plt.gca().axvline(np.log10(bfp['pl_during_nh']*1.e22), color='#ff7f0e')
    plt.gca().axvline(np.log10(bfp['bb_during_nh_low']*1.e22), color='#1f77b4', alpha=0.5)
    plt.gca().axvline(np.log10(bfp['pl_during_nh_low']*1.e22), color='#ff7f0e', alpha=0.5)
    plt.gca().axvline(np.log10(bfp['bb_during_nh_up']*1.e22), color='#1f77b4', alpha=0.5)
    plt.gca().axvline(np.log10(bfp['pl_during_nh_up']*1.e22), color='#ff7f0e', alpha=0.5)
    plt.xlabel('$\log_{10}(N_\mathrm{H})$')
    plt.savefig(oi + '_hist_nh.pdf', bbox_inches='tight', pad_inches=0.1, dpi=300)
    
    plt.figure(figsize=(5,3.75))
    plt.hist(bbp[:,1], zg, normed=True, alpha=0.3)
    plt.xscale('log')
    plt.title(oi)
    plt.xlabel('$z$')
    plt.gca().axvline(bfp['bb_during_zz'], color='#1f77b4')
    plt.gca().axvline(bfp['bb_during_zz_low'], color='#1f77b4', alpha=0.5)
    plt.gca().axvline(bfp['bb_during_zz_up'], color='#1f77b4', alpha=0.5)
    plt.savefig(oi + '_hist_z.pdf', bbox_inches='tight', pad_inches=0.1, dpi=300)
    
    plt.figure(figsize=(5,3.75))
    plt.hist(bbp[:,2], tg, normed=True, alpha=0.3)
    plt.hist(bbp[:,6], tg, normed=True, alpha=0.3)
    plt.hist(bbp[:,10], tg, normed=True, alpha=0.3)
    plt.title(oi)
    plt.legend(['Entire', 'First', 'Second'])
    plt.gca().axvline(bfp['bb_during_tt'], color='#1f77b4')
    plt.gca().axvline(bfp['bb_first_tt'], color='#ff7f0e')
    plt.gca().axvline(bfp['bb_second_tt'], color='#2ca02c')
    plt.gca().axvline(bfp['bb_during_tt_low'], color='#1f77b4', alpha=0.5)
    plt.gca().axvline(bfp['bb_first_tt_low'], color='#ff7f0e', alpha=0.5)
    plt.gca().axvline(bfp['bb_second_tt_low'], color='#2ca02c', alpha=0.5)
    plt.gca().axvline(bfp['bb_during_tt_up'], color='#1f77b4', alpha=0.5)
    plt.gca().axvline(bfp['bb_first_tt_up'], color='#ff7f0e', alpha=0.5)
    plt.gca().axvline(bfp['bb_second_tt_up'], color='#2ca02c', alpha=0.5)
    plt.xlabel('$T$ (keV)')
    plt.xlim([0, np.min((1.5, 1.3*np.max((bbp[:,2].max(), bbp[:,6].max(), bbp[:,10].max()))))])
    plt.savefig(oi + '_hist_bb_t.pdf', bbox_inches='tight', pad_inches=0.1, dpi=300)

    plt.figure(figsize=(5,3.75))
    plt.hist(plp[:,1], gg, normed=True, alpha=0.3)
    plt.hist(plp[:,5], gg, normed=True, alpha=0.3)
    plt.hist(plp[:,9], gg, normed=True, alpha=0.3)
    plt.title(oi)
    plt.legend(['Entire', 'First', 'Second'])
    plt.gca().axvline(bfp['pl_during_gg'], color='#1f77b4')
    plt.gca().axvline(bfp['pl_first_gg'], color='#ff7f0e')
    plt.gca().axvline(bfp['pl_second_gg'], color='#2ca02c')
    plt.gca().axvline(bfp['pl_during_gg_low'], color='#1f77b4', alpha=0.5)
    plt.gca().axvline(bfp['pl_first_gg_low'], color='#ff7f0e', alpha=0.5)
    plt.gca().axvline(bfp['pl_second_gg_low'], color='#2ca02c', alpha=0.5)
    plt.gca().axvline(bfp['pl_during_gg_up'], color='#1f77b4', alpha=0.5)
    plt.gca().axvline(bfp['pl_first_gg_up'], color='#ff7f0e', alpha=0.5)
    plt.gca().axvline(bfp['pl_second_gg_up'], color='#2ca02c', alpha=0.5)
    plt.xlabel('$\Gamma$')
    plt.savefig(oi + '_hist_pl_g.pdf', bbox_inches='tight', pad_inches=0.1, dpi=300)

    plt.figure(figsize=(5,3.75))
    plt.hist(bbp[:,3], gofg, normed=True, alpha=0.3)
    plt.hist(bbp[:,7], gofg, normed=True, alpha=0.3)
    plt.hist(bbp[:,11], gofg, normed=True, alpha=0.3)
    plt.gca().axvline(bfp['bb_during_gof'], color='#1f77b4')
    plt.gca().axvline(bfp['bb_first_gof'], color='#ff7f0e')
    plt.gca().axvline(bfp['bb_second_gof'], color='#2ca02c')
    plt.title(oi + ' BB')
    gofd = (bbp[:,3]<bfp['bb_during_gof']).sum()/bbp[:,3].size
    goff = (bbp[:,7]<bfp['bb_first_gof']).sum()/bbp[:,3].size
    gofs = (bbp[:,11]<bfp['bb_second_gof']).sum()/bbp[:,3].size
    plt.legend(['Entire {0:.2f}'.format(gofd), 'First {0:.2f}'.format(goff), 'Second {0:.2f}'.format(gofs)])
    plt.xlabel('Cram\\\'{e}r--von Mises')
    plt.savefig(oi + '_hist_bb_cvm.pdf', bbox_inches='tight', pad_inches=0.1, dpi=300)

    plt.figure(figsize=(5,3.75))
    plt.hist(plp[:,3], gofg, normed=True, alpha=0.3)
    plt.hist(plp[:,7], gofg, normed=True, alpha=0.3)
    plt.hist(plp[:,11], gofg, normed=True, alpha=0.3)
    plt.gca().axvline(bfp['pl_during_gof'], color='#1f77b4')
    plt.gca().axvline(bfp['pl_first_gof'], color='#ff7f0e')
    plt.gca().axvline(bfp['pl_second_gof'], color='#2ca02c')
    plt.title(oi + ' PL')
    gofd = (plp[:,3]<bfp['pl_during_gof']).sum()/plp[:,3].size
    goff = (plp[:,7]<bfp['pl_first_gof']).sum()/plp[:,3].size
    gofs = (plp[:,11]<bfp['pl_second_gof']).sum()/plp[:,3].size
    plt.legend(['Entire {0:.2f}'.format(gofd), 'First {0:.2f}'.format(goff), 'Second {0:.2f}'.format(gofs)])
    plt.xlabel('Cram\\\'{e}r--von Mises')
    plt.savefig(oi + '_hist_pl_cvm.pdf', bbox_inches='tight', pad_inches=0.1, dpi=300)

    # plt.show()
    plt.close('all')
db()
