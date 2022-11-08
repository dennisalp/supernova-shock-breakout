'''
2020-01-24, Dennis Alp, dalp@kth.se

Compute some parameters used for implementing the model of sapir13 and waxman17b into XSPEC.
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




################################################################
# Parameters
XX = 0.7
YY = 0.3
MM = 15*Msun
RR = 40*Rsun # menon19, utrobin19, alp19
LL = 150000*Lsun # woosley87 (6.e38 erg)
Eexp = 1.5e51
rho = 1 # 1.e-9 g cm-3 waxman17b Eq. 50

AA = np.nan # atomic weight A = ΣixiAi, where xi are the atomic fractions of each element in the envelope compo- sition. Section 2.2 sapir13
nn = np.nan # ion number density, np,0 = ρ0 /(Amp ), of np,0 = 1015A−1 cm−3 Section 3.1 sapir13 NOTE THAT A IS INCLUDED IN NP
mu = (2*XX+0.75*YY)**-1
kk = (1+XX)/5. # The total opacity is then κ = (Σi xi Zi /Σi xi Ai )(σT / mp ). Section 2.3 sapir13
vej = np.sqrt(Eexp/MM)

# "fp is a numerical factor of order unity that depends on the detailed envelope structure" Section 2 waxman17b (from eq. 37 sapir13)
fp = 0.072*(mu/0.62)**4*(LL/(1.e5*Lsun))**-1*(MM/(10*Msun))**3*((1+XX)/1.7)**-1*(1.35-0.35*(LL/(1.e5*Lsun))*(MM/(10*Msun))**-1*((1+XX)/1.7))**4
vbo = 13*vej*(MM/(10*Msun))**0.16*(vej/3.e8)**0.16*(RR/1.e12)**-0.32*(kk/0.34)**0.16*fp**-0.05
rbo = 8.e-9*(MM/(10*Msun))**0.13*(vej/3.e8)**-0.87*(RR/1.e12)**-1.26*(kk/0.34)**-0.87*fp**0.29
TT = 10**(1.4+(vbo/1.e9)**0.5+(0.25-0.05*(vbo/1.e9)**0.5)*np.log10(rbo/1.e-9))/3.e3

# The initial density profile is assumed to be a power law of the distance from the surface, ρ ∝ xn. Section 2 sapir13
# waxman17b, n = 3 (appropriate for a blue supergiant (BSG)) (for radiative envelopes Section 2.1.1 levinson19
Rcoef = 2.2e47*(RR/1.e13)**2*(vbo/1.e9)*(kk/0.34)**-1

db()

'''
For CSM, Section 1.1 levinson19

A different signal is expected when the progenitor is surrounded by a wind. If the wind is thick enough to sustain an RMS then the breakout can take place at a radius much larger than R∗. The duration of the breakout signal is significantly longer, ≈ Rbo/vsh, and the energy it releases is considerably larger.

Section 6.1 waxman17b
The characteristic duration of the pulse is tbo􏰁􏰄Rbo=c􏰁Rbo=vbo

Section 2.1 svirski14
Observationally tbo is also roughly the rise time of the breakout
pulse.
'''
