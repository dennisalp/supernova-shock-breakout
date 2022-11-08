'''
Take an XMM Observation ID and return the primary science target.
'''

import sys
import numpy as np
from astropy.io import fits

obs_list = fits.open('/Users/silver/Box Sync/phd/pro/sne/sbo/3xmm_dr8/3xmmdr8_obslist.fits')[1].data

print(obs_list['TARGET'][np.argmax(obs_list['OBS_ID'] == sys.argv[1])])
