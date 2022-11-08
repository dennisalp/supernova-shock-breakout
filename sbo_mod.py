'''
2020-04-05, Dennis Alp, dalp@kth.se

Using the SBO model (sapir13, waxman17b) to infer SN parameters from observables.
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
from scipy.optimize import fsolve

#For LaTeX style font in plots
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rc('text', usetex=True)






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





################################################################
# Parameters
XX = 0.7
YY = 1-XX
MM = 15*Msun
LL = 150000*Lsun # woosley87 (6.e38 erg)

mu = (2*XX+0.75*YY)**-1
kk = (1+XX)/5. # The total opacity is then κ = (Σi xi Zi /Σi xi Ai )(σT / mp ). Section 2.3 sapir13

# "fp is a numerical factor of order unity that depends on the detailed envelope structure" Section 2 waxman17b (from eq. 37 sapir13)
fp = 0.072*(mu/0.62)**4*(LL/(1.e5*Lsun))**-1*(MM/(10*Msun))**3*((1+XX)/1.7)**-1*(1.35-0.35*(LL/(1.e5*Lsun))*(MM/(10*Msun))**-1*((1+XX)/1.7))**4


def old(EE, TT):
    # Help functions
    def eq_EE(RR, vbo):
        return np.log10(2.2e47*(RR/1.e13)**2*(vbo/1.e9)*(kk/0.34)**-1)-np.log10(EE)
    
    def eq_vbo(vbo, vej, RR):
        tmp = 13*(MM/(10*Msun))**0.16*(vej/3.e8)**0.16*(RR/1.e12)**-0.32*(kk/0.34)**0.16*fp**-0.05
        return tmp-vbo/vej
    
    def eq_rbo(rbo, vej, RR):
        tmp = 8.e-9*(MM/(10*Msun))**0.13*(vej/3.e8)**-0.87*(RR/1.e12)**-1.26*(kk/0.34)**-0.87*fp**0.29
        return np.log10(tmp)-np.log10(rbo)
    
    def eq_TT(vbo, rbo):
        return 1.4+(vbo/1.e9)**0.5+(0.25-0.05*(vbo/1.e9)**0.5)*np.log10(rbo/1.e-9)-np.log10(3*TT*1.e3)



    # Main equation system
    def eqs(pars):
        RR, vbo, vej, rbo = np.power(10, pars)
        print('R: {0:.2e}, vbo: {1:.2e}, vej: {2:.2e}, rho: {3:.2e}'.format(RR, vbo, vej, rbo))
        # print('R: {0:.2e}, vbo: {1:.2e}, vej: {2:.2e}, rho: {3:.2e}'.format(eq_EE(RR, vbo), eq_vbo(vbo, vej, RR), eq_rbo(rbo, vej, RR), eq_TT(vbo, rbo)))
        print()
        return (eq_EE(RR, vbo), eq_vbo(vbo, vej, RR), eq_rbo(rbo, vej, RR), eq_TT(vbo, rbo))

    

    guess = (40*Rsun, 2346980064.7923965, 224224264.6654375, 1.7495697061156578e-09)
    RR, vbo, vej, rbo = fsolve(eqs, np.log10(guess))
    print(eqs((RR, vbo, vej, rbo)))



# A solution
RR, vbo, vej, rbo = (40*Rsun, 2346980064.7923965, 224224264.6654375, 1.7495697061156578e-09)
EE = 3.9984905221170706e+46
TT = 0.3140428131229593

vbo - 13*(MM/(10*Msun))**0.16*vej*(vej/3.e8)**0.16*(RR/1.e12)**-0.32*(kk/0.34)**0.16*fp**-0.05
vbo - 13*(MM/(10*Msun))**0.16*(kk/0.34)**0.16*fp**-0.05*vej*(vej/3.e8)**0.16*(RR/1.e12)**-0.32 # reorder
vbo - 15.215674671871444*vej*(vej/3.e8)**0.16*(RR/1.e12)**-0.32 # compute numerical factor
vbo - 15.215674671871444*vej*(vej/3.e8)**0.16*(np.sqrt(EE*(kk/0.34)/(2.2e45*(vbo/1.e9))))**-0.32 # insert RR
vbo - 15.215674671871444*vej*(vej/3.e8)**0.16*(EE*(kk/0.34)/(2.2e45*(vbo/1.e9)))**-0.16 # simplify
vbo - 273576533.90121686*vej*(vej/3.e8)**0.16*((vbo/1.e9)/EE)**0.16 # compute constat, 15.215674671871444*2.2e45**0.16
vbo - 437259.6896189273*vej**1.16*(vbo/EE)**0.16 # compute constat, 273576533.90121686*1/3.e17**0.16
vej - 1.3718087995461164e-05*vbo**0.7241379310344828*EE**0.13793103448275865 # solve for vej, 1/437259.6896189273**(1/1.16)
vej - (13*(MM/(10*Msun))**0.16*(kk/0.34)**0.16*fp**-0.05*2.2e45**0.16*1/3.e17**0.16)**(-1/1.16)*vbo**0.7241379310344828*EE**0.13793103448275865 # Analytical

rbo - 8.e-9*(MM/(10*Msun))**0.13*(vej/3.e8)**-0.87*(RR/1.e12)**-1.26*(kk/0.34)**-0.87*fp**0.29
rbo - 8.e-9*(MM/(10*Msun))**0.13*(kk/0.34)**-0.87*fp**0.29*(vej/3.e8)**-0.87*(RR/1.e12)**-1.26 # reorder
rbo - 4.931476197786381e-09*(vej/3.e8)**-0.87*(RR/1.e12)**-1.26 # compute numerical factor
rbo - 4.931476197786381e-09*(1.3718087995461164e-05*vbo**0.7241379310344828*EE**0.13793103448275865/3.e8)**-0.87*(RR/1.e12)**-1.26 # insert vej
rbo - 1988.9644214086152*vbo**-0.63*EE**-0.12*(RR/1.e12)**-1.26 # compute numerical factor, 4.931476197786381e-09*(1.3718087995461164e-05/3.e8)**-0.87
rbo - 1988.9644214086152*vbo**-0.63*EE**-0.12*(np.sqrt(EE*(kk/0.34)/(2.2e45*(vbo/1.e9))))**-1.26 # insert RR
rbo - 1988.9644214086152*vbo**-0.63*EE**-0.12*(EE/(2.2e36*vbo))**-0.63 # simplify
rbo - 1988.9644214086152*vbo**-0.63*EE**-0.12*(2.2e36*vbo/EE)**0.63 # simplify
rbo - 1.564419798242583e+26*EE**-0.75 # simplify 1988.9644214086152*2.2e36**0.63
rbo - 4.9471297791076424e-09*(EE/1.e46)**-0.75
rbo - 8.e-9*(MM/(10*Msun))**0.13*(kk/0.34)**-0.87*fp**0.29*(((13*(MM/(10*Msun))**0.16*(kk/0.34)**0.16*fp**-0.05*2.2e45**0.16*1/3.e17**0.16)**(-1/1.16))/3.e8)**-0.87*2.2e36**0.63*EE**-0.75 # Analytical

TT - 10**(1.4+(vbo/1.e9)**0.5+(0.25-0.05*(vbo/1.e9)**0.5)*np.log10(rbo/1.e-9))/3.e3

RR/1.e12 - np.sqrt(EE*(kk/0.34)/(2.2e45*(vbo/1.e9)))
vej - np.sqrt(1.5e51/MM)
1.5e51 - MM*vej**2

def get_sbo_par(EE, TT):
    mu = (2*XX+0.75*YY)**-1
    kk = (1+XX)/5. # The total opacity is then κ = (Σi xi Zi /Σi xi Ai )(σT / mp ). Section 2.3 sapir13
    fp = 0.072*(mu/0.62)**4*(LL/(1.e5*Lsun))**-1*(MM/(10*Msun))**3*((1+XX)/1.7)**-1*(1.35-0.35*(LL/(1.e5*Lsun))*(MM/(10*Msun))**-1*((1+XX)/1.7))**4
    print('fp', fp)

    rbo = 1.564419798242583e+26*EE**-0.75
    rbo = 8.e-9*(MM/(10*Msun))**0.13*(kk/0.34)**-0.87*fp**0.29*(((13*(MM/(10*Msun))**0.16*(kk/0.34)**0.16*fp**-0.05*2.2e45**0.16*1/3.e17**0.16)**(-1/1.16))/3.e8)**-0.87*2.2e36**0.63*EE**-0.75 # Analytical

    def hlp(vbo):
        print(vbo)
        return TT - 10**(1.4+(vbo/1.e9)**0.5+(0.25-0.05*(vbo/1.e9)**0.5)*np.log10(rbo/1.e-9))/3.e3

    vbo = fsolve(hlp, 1e9)[0]

    RR = np.sqrt(EE*(kk/0.34)/(2.2e45*(vbo/1.e9)))*1e12

    vej = 1.3718087995461164e-05*vbo**0.7241379310344828*EE**0.13793103448275865
    vej = (13*(MM/(10*Msun))**0.16*(kk/0.34)**0.16*fp**-0.05*2.2e45**0.16*1/3.e17**0.16)**(-1/1.16)*vbo**0.7241379310344828*EE**0.13793103448275865 # Analytical

    Eexp = MM*vej**2
    
    return RR, vbo, vej, rbo, Eexp



def print_par(RR, vbo, vej, rbo, Eexp):
    print(RR/Rsun, 'Rsun')
    print(vbo/1.e8, '1.e3 km s-1')
    print(vej/1.e8, '1.e3 km s-1')
    print(rbo, 'g cm-3')
    print(Eexp, 'erg')

if __name__ == '__main__':
    print_par(*get_sbo_par(EE, TT))
    XX = 0.
    YY = 1-XX
    MM = 1.5*Msun
    LL = 1.5e6*Lsun # woosley87 (6.e38 erg)
    print_par(*get_sbo_par(EE, TT))
    # fp 0.15722977585234774
    # 39.99999999999999 Rsun
    # 23.46980064792397 km s-1
    # 2.2422426466543794 km s-1
    # 1.7495697061156603e-09 g cm-3
    # 1.500000000000006e+51 erg
    # fp 80.68876210054951
    # 33.050855428953035 Rsun
    # 20.221578471575988 km s-1
    # 3.8938068846926184 km s-1
    # 7.081303384861891e-09 g cm-3
    # 4.523502758692678e+50 erg
    db()



'''
The initial density profile is assumed to be a power law of the distance from the surface, ρ ∝ xn. Section 2 sapir13
waxman17b, n = 3 (appropriate for a blue supergiant (BSG)) (for radiative envelopes Section 2.1.1 levinson19

For CSM, Section 1.1 levinson19

A different signal is expected when the progenitor is surrounded by a wind. If the wind is thick enough to sustain an RMS then the breakout can take place at a radius much larger than R∗. The duration of the breakout signal is significantly longer, ≈ Rbo/vsh, and the energy it releases is considerably larger.

Section 6.1 waxman17b
The characteristic duration of the pulse is tbo􏰁􏰄Rbo=c􏰁Rbo=vbo

Section 2.1 svirski14
Observationally tbo is also roughly the rise time of the breakout
pulse.
'''

