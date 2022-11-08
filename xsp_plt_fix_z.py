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
    # for kk in sorted(bfp.keys()): print(kk + '\t', bfp[kk])
    bfp = {}

    ff = open(oi + '_fit_bb_during.log')
    seen = False
    for line in reversed(list(ff)):
        if line[12:25] == 'zTBabs     nH':
            bfp['bb_during_nh'] = float(line.split()[5])
            break
        elif line[12:30] == 'clumin     lg10Lum':
            bfp['bb_during_ll'] = float(line.split()[5])
        elif line[12:25] == 'zbbody     kT':
            bfp['bb_during_tt'] = float(line.split()[5])
        elif line[:29] == 'Fit statistic : C-Statistic =':
            bfp['bb_during_gof'] = float(line.split()[5])
            bfp['bb_during_dof'] = int(line.split()[11])

        if line[:7] == '     2 ' and 'bb_during_nh_lim' in bfp.keys() and seen:
            bfp['bb_during_nh_low'] = float(line.split()[1])
            bfp['bb_during_nh_upp'] = float(line.split()[2])
        elif line[:7] == '     2 ':
            bfp['bb_during_nh_lim'] = float(line.split()[2])
            bfp['bb_during_nh_li2'] = float(line.split()[1])
            
        elif line[:7] == '     7 ' and 'bb_during_ll_lim' in bfp.keys() and seen:
            bfp['bb_during_ll_low'] = float(line.split()[1])
            bfp['bb_during_ll_upp'] = float(line.split()[2])
        elif line[:7] == '     7 ':
            bfp['bb_during_ll_lim'] = float(line.split()[2])
            bfp['bb_during_ll_li2'] = float(line.split()[1])
            
        elif line[:7] == '     8 ' and 'bb_during_tt_lim' in bfp.keys() and seen:
            bfp['bb_during_tt_low'] = float(line.split()[1])
            bfp['bb_during_tt_upp'] = float(line.split()[2])
        elif line[:7] == '     8 ':
            bfp['bb_during_tt_lim'] = float(line.split()[2])
            bfp['bb_during_tt_li2'] = float(line.split()[1])
        elif line[:13] == 'XSPEC12>error':
            seen = True


            
    ff = open(oi + '_fit_bb_first.log')
    seen = False
    for line in reversed(list(ff)):
        if line[12:25] == 'zbbody     kT':
            bfp['bb_first_tt'] = float(line.split()[5])
            break
        elif line[:29] == 'Fit statistic : C-Statistic =':
            bfp['bb_first_gof'] = float(line.split()[5])
            bfp['bb_first_dof'] = int(line.split()[11])
            
        if line[:7] == '     8 ' and 'bb_first_tt_lim' in bfp.keys() and seen:
            bfp['bb_first_tt_low'] = float(line.split()[1])
            bfp['bb_first_tt_upp'] = float(line.split()[2])
        elif line[:7] == '     8 ':
            bfp['bb_first_tt_lim'] = float(line.split()[2])
            bfp['bb_first_tt_li2'] = float(line.split()[1])
        elif line[:13] == 'XSPEC12>error':
            seen = True


            
    ff = open(oi + '_fit_bb_second.log')
    seen = False
    for line in reversed(list(ff)):
        if line[12:25] == 'zbbody     kT':
            bfp['bb_second_tt'] = float(line.split()[5])
            break
        elif line[:29] == 'Fit statistic : C-Statistic =':
            bfp['bb_second_gof'] = float(line.split()[5])
            bfp['bb_second_dof'] = int(line.split()[11])

        if line[:7] == '     8 ' and 'bb_second_tt_lim' in bfp.keys() and seen:
            bfp['bb_second_tt_low'] = float(line.split()[1])
            bfp['bb_second_tt_upp'] = float(line.split()[2])
        elif line[:7] == '     8 ':
            bfp['bb_second_tt_lim'] = float(line.split()[2])
            bfp['bb_second_tt_li2'] = float(line.split()[1])
        elif line[:13] == 'XSPEC12>error':
            seen = True





    # Power law
    ff = open(oi + '_fit_pl_during.log')
    seen = False
    for line in reversed(list(ff)):
        if line[12:25] == 'zTBabs     nH':
            bfp['pl_during_nh'] = float(line.split()[5])
            break
        elif line[12:31] == 'zpowerlw   PhoIndex':
            bfp['pl_during_gg'] = float(line.split()[4])
        elif line[12:30] == 'clumin     lg10Lum':
            bfp['pl_during_ll'] = float(line.split()[5])
        elif line[:29] == 'Fit statistic : C-Statistic =':
            bfp['pl_during_gof'] = float(line.split()[5])
            bfp['pl_during_dof'] = int(line.split()[11])
        
        if line[:7] == '     2 ' and 'pl_during_nh_lim' in bfp.keys() and seen:
            bfp['pl_during_nh_low'] = float(line.split()[1])
            bfp['pl_during_nh_upp'] = float(line.split()[2])
        elif line[:7] == '     2 ':
            bfp['pl_during_nh_lim'] = float(line.split()[2])
            bfp['pl_during_nh_li2'] = float(line.split()[1])
        elif line[:7] == '     8 ' and 'pl_during_gg_lim' in bfp.keys() and seen:
            bfp['pl_during_gg_low'] = float(line.split()[1])
            bfp['pl_during_gg_upp'] = float(line.split()[2])
        elif line[:7] == '     8 ':
            bfp['pl_during_gg_lim'] = float(line.split()[2])
            bfp['pl_during_gg_li2'] = float(line.split()[1])
        elif line[:7] == '     7 ' and 'pl_during_ll_lim' in bfp.keys() and seen:
            bfp['pl_during_ll_low'] = float(line.split()[1])
            bfp['pl_during_ll_upp'] = float(line.split()[2])
        elif line[:7] == '     7 ':
            bfp['pl_during_ll_lim'] = float(line.split()[2])
            bfp['pl_during_ll_li2'] = float(line.split()[1])
        elif line[:13] == 'XSPEC12>error':
            seen = True

    ff = open(oi + '_fit_pl_first.log')
    seen = False
    for line in reversed(list(ff)):
        if line[12:31] == 'zpowerlw   PhoIndex':
            bfp['pl_first_gg'] = float(line.split()[4])
            break
        elif line[:29] == 'Fit statistic : C-Statistic =':
            bfp['pl_first_gof'] = float(line.split()[5])
            bfp['pl_first_dof'] = int(line.split()[11])

        if line[:7] == '     8 ' and 'pl_first_gg_lim' in bfp.keys() and seen:
            bfp['pl_first_gg_low'] = float(line.split()[1])
            bfp['pl_first_gg_upp'] = float(line.split()[2])
        elif line[:7] == '     8 ':
            bfp['pl_first_gg_lim'] = float(line.split()[2])
            bfp['pl_first_gg_li2'] = float(line.split()[1])
        elif line[:13] == 'XSPEC12>error':
            seen = True

            
    ff = open(oi + '_fit_pl_second.log')
    seen = False
    for line in reversed(list(ff)):
        if line[12:31] == 'zpowerlw   PhoIndex':
            bfp['pl_second_gg'] = float(line.split()[4])
            break
        elif line[:29] == 'Fit statistic : C-Statistic =':
            bfp['pl_second_gof'] = float(line.split()[5])
            bfp['pl_second_dof'] = int(line.split()[11])

        if line[:7] == '     8 ' and 'pl_second_gg_lim' in bfp.keys() and seen:
            bfp['pl_second_gg_low'] = float(line.split()[1])
            bfp['pl_second_gg_upp'] = float(line.split()[2])
        elif line[:7] == '     8 ':
            bfp['pl_second_gg_lim'] = float(line.split()[2])
            bfp['pl_second_gg_li2'] = float(line.split()[1])
        elif line[:13] == 'XSPEC12>error':
            seen = True
            
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
        elif line[:29] == 'Fit statistic : C-Statistic =':
            gof = float(line.split()[5])
            

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
        elif line[:29] == 'Fit statistic : C-Statistic =':
            gof = float(line.split()[5])

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
    upp = bfp[lab+'_upp']
    lim = bfp[lab+'_lim']
    li2 = bfp[lab+'_li2']

    if 'nh' in lab or 'gg' in lab: dig = '1'
    else: dig = '2'
    if np.abs(low) < 1.e-6 and np.abs(lim) < 1.e-6:
        return '\\nodata{}'
    elif np.abs(upp) < 1.e-6 and np.abs(li2) < 1.e-6:
        return '\\nodata{}'
    elif np.abs(low) < 1.e-6:
        return ('$<{0:.'+dig+'f}$').format(lim)
    elif np.abs(upp) < 1.e-6:
        return ('$>{0:.'+dig+'f}$').format(li2)
    else:
        return ('${0:.'+dig+'f}_{{{1:.'+dig+'f}}}^{{+{2:.'+dig+'f}}}$').format(val, low-val, upp-val)




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
# obs = ['0149780101']

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




################################################################
# Parse the MC logs
buf  = '    \\colhead{XT} & \\colhead{$N_\\mathrm{H,MW}$}     & \\colhead{$N_\\mathrm{H,bb}$}     & \\colhead{$T$}          & \\colhead{$T_1$}        & \\colhead{$T_2$}        & \\colhead{$N_\\mathrm{H,pl}$}     & \\colhead{$\\Gamma$}     & \\colhead{$\\Gamma_1$}   & \\colhead{$\\Gamma_2$}   \\\\\n'
buf += '    \\colhead{}   & \\colhead{($10^{22}$~cm$^{-2}$)} & \\colhead{($10^{22}$~cm$^{-2}$)} & \\colhead{(keV)}        & \\colhead{(keV)}        & \\colhead{(keV)}        & \\colhead{($10^{22}$~cm$^{-2}$)} & \\colhead{}             & \\colhead{}             & \\colhead{}} \\startdata\n'

app  = '      \\colhead{XT} & \\colhead{$C_\\mathrm{BB}/$dof} & \\colhead{$G_\\mathrm{BB}$} & \\colhead{$C_\\mathrm{BB,1}/$dof} & \\colhead{$G_\\mathrm{BB,1}$} & \\colhead{$C_\\mathrm{BB,2}/$dof} & \\colhead{$G_\\mathrm{BB,2}$} & \\colhead{$C_\\mathrm{PL}/$dof} & \\colhead{$G_\\mathrm{PL}$} & \\colhead{$C_\\mathrm{PL,1}/$dof} & \\colhead{$G_\\mathrm{PL,1}$} & \\colhead{$C_\\mathrm{PL,2}/$dof} & \\colhead{$G_\\mathrm{PL,2}$}} \\startdata\n'

gofs = {}

for oi in obs[:]:

    os.chdir('/Users/silver/dat/xmm/sbo/' + oi + '_repro/')
    bfp = get_bf()
    # Get N_H
    ff = open('/Users/silver/box/phd/pro/sne/sbo/src/xsp_mc.sh')
    trigger = False
    for line in ff:
        if 'oi=' + oi in line:
            trigger = True
        elif trigger:
            nh = float(line.strip().split('=')[1])
            break


    
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
    buf += '  {9:15s}& {0:32s}& {1:32s}& {2:23s}& {3:23s}& {4:23s}& {5:32s}& {6:23s}& {7:23s}& {8:23s}\\\\\n'.format(
        '${0:.3f}$'.format(nh), fmt_val('bb_during_nh'), fmt_val('bb_during_tt'), fmt_val('bb_first_tt'), fmt_val('bb_second_tt'),
        fmt_val('pl_during_nh'), fmt_val('pl_during_gg'), fmt_val('pl_first_gg'), fmt_val('pl_second_gg'), xrt[oi])
    
    ################################################################
    # Plots
    os.chdir('/Users/silver/dat/xmm/sbo/' + oi + '_repro/')
    ng = np.linspace(0, 25, 100)
    zg = np.logspace(-3, 1, 100)
    lg = np.linspace(38, 48, 100)
    gg = np.linspace(-1, 8, 100)
    gofg = np.linspace(0, 300, 100)
    tg = np.logspace(np.log10(0.03), np.log10(3), 100)
    
    # plt.figure(figsize=(5,3.75))
    # plt.hist(np.log10(bbp[:,0]*1.e22), ng, normed=True, alpha=0.3)
    # plt.hist(np.log10(plp[:,0]*1.e22), ng, normed=True, alpha=0.3)
    # plt.title(oi)
    # plt.legend(['bb', 'pl'])
    # plt.gca().axvline(np.log10(bfp['bb_during_nh']*1.e22), color='#1f77b4')
    # plt.gca().axvline(np.log10(bfp['pl_during_nh']*1.e22), color='#ff7f0e')
    # plt.gca().axvline(np.log10(bfp['bb_during_nh_low']*1.e22), color='#1f77b4', alpha=0.5)
    # plt.gca().axvline(np.log10(bfp['pl_during_nh_low']*1.e22), color='#ff7f0e', alpha=0.5)
    # plt.gca().axvline(np.log10(bfp['bb_during_nh_upp']*1.e22), color='#1f77b4', alpha=0.5)
    # plt.gca().axvline(np.log10(bfp['pl_during_nh_upp']*1.e22), color='#ff7f0e', alpha=0.5)
    # plt.xlabel('$\log_{10}(N_\mathrm{H})$')
    # plt.savefig(oi + '_hist_nh.pdf', bbox_inches='tight', pad_inches=0.1, dpi=300)
    
    # plt.figure(figsize=(5,3.75))
    # plt.hist(bbp[:,1], zg, normed=True, alpha=0.3)
    # plt.xscale('log')
    # plt.title(oi)
    # plt.xlabel('$z$')
    # plt.gca().axvline(bfp['bb_during_ll'], color='#1f77b4')
    # plt.gca().axvline(bfp['bb_during_ll_low'], color='#1f77b4', alpha=0.5)
    # plt.gca().axvline(bfp['bb_during_ll_upp'], color='#1f77b4', alpha=0.5)
    # plt.savefig(oi + '_hist_z.pdf', bbox_inches='tight', pad_inches=0.1, dpi=300)
    
    # plt.figure(figsize=(5,3.75))
    # plt.hist(bbp[:,2], tg, normed=True, alpha=0.3)
    # plt.hist(bbp[:,6], tg, normed=True, alpha=0.3)
    # plt.hist(bbp[:,10], tg, normed=True, alpha=0.3)
    # plt.title(oi)
    # plt.legend(['Entire', 'First', 'Second'])
    # plt.gca().axvline(bfp['bb_during_tt'], color='#1f77b4')
    # plt.gca().axvline(bfp['bb_first_tt'], color='#ff7f0e')
    # plt.gca().axvline(bfp['bb_second_tt'], color='#2ca02c')
    # plt.gca().axvline(bfp['bb_during_tt_low'], color='#1f77b4', alpha=0.5)
    # plt.gca().axvline(bfp['bb_first_tt_low'], color='#ff7f0e', alpha=0.5)
    # plt.gca().axvline(bfp['bb_second_tt_low'], color='#2ca02c', alpha=0.5)
    # plt.gca().axvline(bfp['bb_during_tt_upp'], color='#1f77b4', alpha=0.5)
    # plt.gca().axvline(bfp['bb_first_tt_upp'], color='#ff7f0e', alpha=0.5)
    # plt.gca().axvline(bfp['bb_second_tt_upp'], color='#2ca02c', alpha=0.5)
    # plt.xlabel('$T$ (keV)')
    # plt.xlim([0, np.min((1.5, 1.3*np.max((bbp[:,2].max(), bbp[:,6].max(), bbp[:,10].max()))))])
    # plt.savefig(oi + '_hist_bb_t.pdf', bbox_inches='tight', pad_inches=0.1, dpi=300)

    # plt.figure(figsize=(5,3.75))
    # plt.hist(plp[:,1], gg, normed=True, alpha=0.3)
    # plt.hist(plp[:,5], gg, normed=True, alpha=0.3)
    # plt.hist(plp[:,9], gg, normed=True, alpha=0.3)
    # plt.title(oi)
    # plt.legend(['Entire', 'First', 'Second'])
    # plt.gca().axvline(bfp['pl_during_gg'], color='#1f77b4')
    # plt.gca().axvline(bfp['pl_first_gg'], color='#ff7f0e')
    # plt.gca().axvline(bfp['pl_second_gg'], color='#2ca02c')
    # plt.gca().axvline(bfp['pl_during_gg_low'], color='#1f77b4', alpha=0.5)
    # plt.gca().axvline(bfp['pl_first_gg_low'], color='#ff7f0e', alpha=0.5)
    # plt.gca().axvline(bfp['pl_second_gg_low'], color='#2ca02c', alpha=0.5)
    # plt.gca().axvline(bfp['pl_during_gg_upp'], color='#1f77b4', alpha=0.5)
    # plt.gca().axvline(bfp['pl_first_gg_upp'], color='#ff7f0e', alpha=0.5)
    # plt.gca().axvline(bfp['pl_second_gg_upp'], color='#2ca02c', alpha=0.5)
    # plt.xlabel('$\Gamma$')
    # plt.savefig(oi + '_hist_pl_g.pdf', bbox_inches='tight', pad_inches=0.1, dpi=300)

    plt.figure(figsize=(5,3.75))
    plt.hist(bbp[:,3], gofg, normed=True, alpha=0.3)
    plt.hist(bbp[:,7], gofg, normed=True, alpha=0.3)
    plt.hist(bbp[:,11], gofg, normed=True, alpha=0.3)
    plt.gca().axvline(bfp['bb_during_gof'], color='#1f77b4')
    plt.gca().axvline(bfp['bb_first_gof'], color='#ff7f0e')
    plt.gca().axvline(bfp['bb_second_gof'], color='#2ca02c')
    plt.title(oi + ' BB')
    bbgofd = (bbp[:,3]<bfp['bb_during_gof']).sum()/bbp[:,3].size
    bbgoff = (bbp[:,7]<bfp['bb_first_gof']).sum()/bbp[:,3].size
    bbgofs = (bbp[:,11]<bfp['bb_second_gof']).sum()/bbp[:,3].size
    plt.legend(['Entire {0:.2f}'.format(bbgofd), 'First {0:.2f}'.format(bbgoff), 'Second {0:.2f}'.format(bbgofs)])
    plt.xlabel('C-stat')
    plt.savefig(oi + '_hist_bb_cst.pdf', bbox_inches='tight', pad_inches=0.1, dpi=300)

    plt.figure(figsize=(5,3.75))
    plt.hist(plp[:,3], gofg, normed=True, alpha=0.3)
    plt.hist(plp[:,7], gofg, normed=True, alpha=0.3)
    plt.hist(plp[:,11], gofg, normed=True, alpha=0.3)
    plt.gca().axvline(bfp['pl_during_gof'], color='#1f77b4')
    plt.gca().axvline(bfp['pl_first_gof'], color='#ff7f0e')
    plt.gca().axvline(bfp['pl_second_gof'], color='#2ca02c')
    plt.title(oi + ' PL')
    plgofd = (plp[:,3]<bfp['pl_during_gof']).sum()/plp[:,3].size
    plgoff = (plp[:,7]<bfp['pl_first_gof']).sum()/plp[:,3].size
    plgofs = (plp[:,11]<bfp['pl_second_gof']).sum()/plp[:,3].size
    plt.legend(['Entire {0:.2f}'.format(plgofd), 'First {0:.2f}'.format(plgoff), 'Second {0:.2f}'.format(plgofs)])
    plt.xlabel('C-stat')
    plt.savefig(oi + '_hist_pl_cst.pdf', bbox_inches='tight', pad_inches=0.1, dpi=300)

    # plt.show()
    plt.close('all')

    def fmt_dof(lab):
        gof = bfp[lab + '_gof']
        dof = bfp[lab + '_dof']
        red = gof/dof
        return  '${0:d}/{1:d}={2:.2f}$'.format(int(np.round(gof)), dof, red)
    app += '    {0:15s}& {1:29s} & {2:25.2f} & {3:31s} & {4:27.2f} & {5:31s} & {6:27.2f} & {7:29s} & {8:25.2f} & {9:31s} & {10:27.2f} & {11:31s} & {12:27.2f} \\\\\n'.format(
        xrt[oi],
        fmt_dof('bb_during'), bbgofd,
        fmt_dof('bb_first'), bbgoff,
        fmt_dof('bb_second'), bbgofs,
        fmt_dof('pl_during'), plgofd,
        fmt_dof('pl_first'), plgoff,
        fmt_dof('pl_second'), plgofs)

    gofs[oi + '_bb'] = bbgofd
    gofs[oi + '_pl'] = plgofd


np.save('/Users/silver/box/phd/pro/sne/sbo/src/meta/gofs.npy', gofs) 

print('\n\n\n\n\n\n')
print(buf)
print('\n\n\n\n\n\n')
print(app)
print('\n\n\n\n\n\n')
db()
