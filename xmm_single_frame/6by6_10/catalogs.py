from pdb import set_trace as st
import time
import numpy as np

from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier
from astroquery.simbad import Simbad
from astroquery.ned import Ned
from astropy import units as u

import warnings
from astroquery.exceptions import BlankResponseWarning
warnings.filterwarnings("ignore", category=BlankResponseWarning, message=".*No astronomical object found.*")



gaia_col = ['_r', 'RA_ICRS', 'DE_ICRS', 'Plx', 'e_Plx', 'Gmag', 'e_Gmag']
mass_col = ['_r', 'RAJ2000', 'DEJ2000', 'Hmag', 'e_Hmag'] 
sdss_col = ['_r', 'RA_ICRS', 'DE_ICRS', 'class', 'rmag', 'e_rmag', 'zsp', 'zph', 'e_zph', '__zph_']
Simbad._VOTABLE_FIELDS = ['distance_result', 'ra', 'dec', 'plx', 'plx_error','rv_value', 'z_value', 'otype(V)']
sdss_type = ['', '', '', 'galaxy', '', '', 'star']
sup_cos = ['', 'galaxy', 'star', 'unclassifiable', 'noise']
bad_types = ['star', 'cv', 'stellar', 'spectroscopic', 'eclipsing', 'variable of', 'binary', 'white', 'cepheid', 'dwarf']
agn_types = ['seyfert', 'quasar', 'blazar', 'bl lac', 'liner', 'active']



def get_viz(coo):
    def have_gaia(tab):
        return (tab['Plx']/tab['e_Plx']).max() > 3

    def viz_hlp(coo, ts):
        # Recursive pauser to avoid blacklisting
        try:
            viz = Vizier(columns=['*', '+_r'], timeout=60, row_limit=10).query_region(coo, radius=5*u.arcsec, catalog='I/345/gaia2,II/246/out,V/147/sdss12')
        except:
            print('Vizier blacklisted, pausing for', ts, 's.')
            time.sleep(ts)
            viz = viz_hlp(coo, min(2*ts, 600))
        return viz

    viz = viz_hlp(coo, 8)
    nearest = 999.
    if len(viz) == 0:
        return '', False, nearest

    # Some formating
    tmp = []
    plx_star = False
    if 'I/345/gaia2' in viz.keys():
        tmp = tmp + ['Gaia DR2'] + viz['I/345/gaia2'][gaia_col].pformat(max_width=-1)
        plx_star = have_gaia(viz['I/345/gaia2'][gaia_col])
        nearest = np.min((viz['I/345/gaia2']['_r'].min(), nearest))
    if 'II/246/out' in viz.keys():
        tmp = tmp + ['\n2MASS'] + viz['II/246/out'][mass_col].pformat(max_width=-1)
        nearest = np.min((viz['II/246/out']['_r'].min(), nearest))
    if 'V/147/sdss12' in viz.keys():
        tmp = tmp + ['\nSDSS12'] + viz['V/147/sdss12'][sdss_col].pformat(max_width=-1)
        nearest = np.min((viz['V/147/sdss12']['_r'].min(), nearest))
    return '\n'.join(tmp), plx_star, nearest

def get_sim(coo):
    def sim_hlp(coo, ts):
        # Recursive pauser to avoid blacklisting
        try:
            sim = Simbad.query_region(coo, radius=60*u.arcsec)
        except:
            print('SIMBAD blacklisted, pausing for', ts, 's.')
            time.sleep(ts)
            sim = sim_hlp(coo, min(2*ts, 600))
        return sim

    sim = sim_hlp(coo, 8)
    if sim is None or len(sim) == 0:
        return '', 999, True, False

    otype_v = sim[0]['OTYPE_V'].lower()
    if sim['DISTANCE_RESULT'][0] < 5.:
        good_type = not any(bt in otype_v for bt in bad_types)
        agn = any(at in otype_v for at in agn_types)
    else:
        good_type = True
        agn = False
    return '\n\nSIMBAD\n' + '\n'.join(sim.pformat(max_lines=-1, max_width=-1)[:13]), sim['DISTANCE_RESULT'].min(), good_type, agn

def get_ned(coo):
    #NED bug, bug in NED
    if coo.separation(SkyCoord(271.13313251264*u.deg, 67.523408125256*u.deg)).arcsec < 1e-6:
        coo = SkyCoord(271.13313251264*u.deg, 67.5235*u.deg)

    def ned_hlp(coo, ts):
        # Recursive pauser to avoid blacklisting
        try:
            ned = Ned.query_region(coo, radius=60*u.arcsec)
        except:
            print('NED blacklisted, pausing for', ts, 's.')
            #NED bug, bug in NED: *** requests.exceptions.ConnectionError: HTTPConnectionPool(host='ned.ipac.caltech.edu', port=80): Read timed out.
            # <SkyCoord (ICRS): (ra, dec) in deg (221.40009467, 68.90898505)>
            # <SkyCoord (ICRS): (ra, dec) in deg (149.59574552, 65.37104883)>
            if ts > 8.1:
                coo = SkyCoord(coo.ra, coo.dec+1e-4*u.deg)
                print('Shifted NED coordinate', 1e-4, 'deg north')
            elif ts > 8.4:
                print('WARNING NED Failed more than 4 times, returning None')
                return None
            time.sleep(ts)
            ned = ned_hlp(coo, ts+0.1)
        return ned

    ned = ned_hlp(coo, 8)
    if ned is None or len(ned) == 0:
        return '', 999

    # Remove 2MASS objects beyond 5 arcsec
    ned = ned['Separation', 'Object Name', 'RA', 'DEC', 'Type', 'Velocity', 'Redshift']
    ned['Separation'].convert_unit_to(u.arcsec)
    rm = []
    for jj, row in enumerate(ned):
        if row['Separation'] > 5. and row['Object Name'][:5] == '2MASS':
            rm.append(jj)
    ned.remove_rows(rm)
    if len(ned) == 0:
        return '', 999

    ned['Separation'] = ned['Separation'].round(3)
    ned['Separation'].unit = u.arcsec
    ned.sort('Separation')
    return '\n\nNED\n' + '\n'.join(ned.pformat(max_lines=-1, max_width=-1)[:13]), ned['Separation'].min()
