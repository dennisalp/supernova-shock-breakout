'''
2019-12-18, Dennis Alp, dalp@kth.se

Simple light curve fitter. Fits linear rise with exponential decay.
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
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.optimize import curve_fit

#For LaTeX style font in plots
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rc('text', usetex=True)



################################################################
oi = sys.argv[1]
cwd = '/Users/silver/dat/xmm/sbo/' + oi + '_repro'
os.chdir(cwd)


ff = open('/Users/silver/box/phd/pro/sne/sbo/src/xmm_' + oi + '.sh')
for line in ff:
    if line[:7] == 't_rise=':
        t_rise = float(line.split('=')[1])
    elif line[:6] == 't_mid=':
        t_mid = float(line.split('=')[1])
    elif line[:7] == 't_fade=':
        t_fade = float(line.split('=')[1])

def lin_lin(tt, t0, tmax, amp, t1):
    res = (tt-t0)/(tmax-t0)*amp
    res = np.where(tt > tmax, (tt-t1)/(tmax-t1)*amp, res)
    res = np.where(tt < t0, 0, res)
    res = np.where(tt > t1, 0, res)
    return res
def lin_exp(tt, t0, tmax, amp, tau):
    res = (tt-t0)/(tmax-t0)*amp
    res = np.where(tt > tmax, amp*np.exp(-(tt-tmax)/tau), res)
    res = np.where(tt < t0, 0, res)
    return res
def lin_linexp(tt, t0, t1, tmax, amp, tau):
    res = (tt-t0)/(tmax-t0)*amp
    res = np.where(tt > tmax, (tt-t1)/(tmax-t1)*amp*np.exp(-(tt-tmax)/tau), res)
    res = np.where(tt < t0, 0, res)
    res = np.where(tt > t1, 0, res)
    return res
def exp_linexp(tt, t1, tmax, amp, tau1, tau2):
    res = amp*np.exp((tt-tmax)/tau1)
    res = np.where(tt > tmax, (tt-t1)/(tmax-t1)*amp*np.exp(-(tt-tmax)/tau2), res)
    res = np.where(tt > t1, 0, res)
    return res
def exp_exp(tt, tmax, amp, tau1, tau2):
    res = amp*np.exp((tt-tmax)/tau1)
    res = np.where(tt > tmax, amp*np.exp(-(tt-tmax)/tau2), res)
    return res
def lin_pow(tt, t0, tmax, amp, gam):
    res = (tt-t0)/(tmax-t0)*amp
    res = np.where(tt > tmax, amp*(tt-t0)**gam/(tmax-t0)**gam, res)
    res = np.where(tt < t0, 0, res)
    return res



################################################################
pn = fits.open(oi + '_pn_lccorr.fits')[1].data
m1 = fits.open(oi + '_m1_lccorr.fits')[1].data
m2 = fits.open(oi + '_m2_lccorr.fits')[1].data

tt = pn['TIME']
rate = pn['RATE']+m1['RATE']+m2['RATE']
error = np.sqrt(pn['ERROR']**2+m1['ERROR']**2+m2['ERROR']**2)
gti = ~np.isnan(error)

tt = tt[gti]
rate = rate[gti]
error = error[gti]
error = np.where(error==0, np.mean(error), error)

# fun = lin_exp
# guess = np.array([t_rise, t_mid, rate.max(), (t_fade-t_mid)/2])
# fun = exp_exp
# guess = np.array([t_mid, rate.max(), (t_mid-t_rise)/2, (t_fade-t_mid)/2])
# fun = lin_pow
# guess = np.array([t_rise, t_mid, rate.max(), -1])
# fun = lin_lin
# guess = np.array([t_rise, t_mid, rate.max(), t_fade])
# fun = lin_linexp
# guess = np.array([t_rise, t_fade, t_mid, rate.max(), (t_fade-t_mid)/2])
fun = exp_linexp
guess = np.array([t_fade, t_mid, rate.max(), (t_mid-t_rise)/2, (t_fade-t_mid)/2])

pars, covar = curve_fit(fun, tt, rate, p0=guess, sigma=error, absolute_sigma=True, method='lm')
perr = np.sqrt(np.diag(covar))



################################################################
tp = np.linspace(tt[0], tt[-1], 100000)
res = fun(tp, *pars)
plt.errorbar(tt-t_rise, rate, yerr=error, label='Sum', fmt='.')
plt.plot(tp-t_rise, res, label='Fit')
# plt.plot(tt-t_rise, pn['RATE'][gti])
# plt.plot(tt-t_rise, m1['RATE'][gti])
# plt.plot(tt-t_rise, m2['RATE'][gti])
plt.legend()
plt.title(oi)
plt.xlabel('$t-' + str(t_rise) + '$ (s)')
plt.ylabel('Rate (cts s$^{-1}$)')
plt.xlim([-60, t_fade-t_rise+60])
plt.ylim([-0.2, 3])
# np.savetxt('${id}_ep_lccorr.txt', np.c_[pn['TIME'], pn['RATE']+m1['RATE']+m2['RATE']])
# plt.savefig('${id}_ep_lccorr.pdf', bbox_inches='tight', pad_inches=0.1, dpi=300)
plt.show()
db()
