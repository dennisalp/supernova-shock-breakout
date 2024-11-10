from pdb import set_trace as st
from glob import glob
import os
import subprocess

from astroquery.esa.xmm_newton import XMMNewton



def read_log(ff):
    # https://heasarc.gsfc.nasa.gov/db-perl/W3Browse/w3table.pl?tablehead=name%3Dxmmmaster&Action=More+Options
    # sort = time
    # time < 2024-08-24  # removes some junk
    # public_date < 2024-08-24
    # data_in_heasarc = Y
    # mos1_mode = *SW* | *FF*
    # mos2_mode = *SW* | *FF*
    # pn_mode = *LW* | *FF*
    # Save All Objects To File
    obs = []
    target = []
    epoch = []

    with open(ff, 'r') as f:
        for ll in f:

            if ll[0] == '#':
                continue
            assert len(ll.split('|')) == 16, 'Pipe (|) used for purpose other than as a delimiter (e.g. in the target name)'
            if ';' in ''.join(ll.split('|')[-4:-1]):
                continue

            obs.append(ll.split('|')[1].strip())
            target.append(ll.split('|')[3].strip())
            epoch.append(ll.split('|')[6].strip())

    return obs, target, epoch



# API and archive connections
# https://nxsa.esac.esa.int/nxsa-web/#astroquery
# https://astroquery.readthedocs.io/en/latest/esa/xmm_newton/xmm_newton.html
# https://nxsa.esac.esa.int/nxsa-web/#aio

# XMM Pipeline info
# https://xmm-tools.cosmos.esa.int/external/xmm_obs_info/odf/data/docs/XMM-SOC-GEN-ICD-0024.pdf
# https://xmm-tools.cosmos.esa.int/external/xmm_user_support/documentation/sas_usg/USG/epicodf.html#4981
# https://xmm-tools.cosmos.esa.int/external/xmm_user_support/documentation/sas_usg/USG.pdf
# https://www.cosmos.esa.int/web/xmm-newton/documentation
# https://www.cosmos.esa.int/web/xmm-newton/pipeline-configurations

# CCD Layouts
# https://www.researchgate.net/figure/MOS1-and-MOS2-CCD-layouts-and-CCD-numbering-small-numbers-on-the-top-right-corner-of-the_fig1_23936243
# https://www.cosmos.esa.int/web/xmm-newton/boundaries-pn
# https://xmm-tools.cosmos.esa.int/external/xmm_user_support/documentation/uhb/moschipgeom.html


def download_data(cwd, oi):
    def dl_hlp(cwd, oi, inst, name, fn):
        try:
            XMMNewton.download_data(oi, level="PPS", instname=inst, name=name, filename=fn, extension='FTZ', verbose=False)
        except Exception as ee:
            print('ERROR Downloading Observation:', oi)
            return False

        gtar = glob('*.tar')
        gftz = glob('*.FTZ')
        if len(gtar) == 1:
            retcode = subprocess.call(['tar', '-xvf', cwd + gtar[0], '-C', cwd], stderr=subprocess.DEVNULL)
            retcode = subprocess.call(['mv'] + glob(cwd + oi + '/pps/*') + [cwd + oi])
            retcode = subprocess.call(['rm', '-rf', cwd + oi + '/pps'])
            retcode = subprocess.call(['rm', cwd + gtar[0]])
            return True
        elif len(gftz) == 1:
            retcode = subprocess.call(['mkdir', '-p', cwd + oi])
            retcode = subprocess.call(['mv', cwd + gftz[0], cwd + oi])
            return True
        
        print('ERROR Unknown data format delivered from XSA AIO:', oi)
        return False

    if os.path.isdir(oi):
        print(oi + ' found on disk.')
        return True

    m1 = dl_hlp(cwd, oi, 'M1', 'MIEVLI', f'{oi}_M1.FTZ')
    m2 = dl_hlp(cwd, oi, 'M2', 'MIEVLI', f'{oi}_M2.FTZ')
    pn = dl_hlp(cwd, oi, 'PN', 'PIEVLI', f'{oi}_PN.FTZ')

    if m1 and m2 and pn:
        return True

    retcode = subprocess.call(['rm', '-rf', cwd + oi])
    retcode = subprocess.call(['rm'] + glob('*.tar'), stderr=subprocess.DEVNULL)
    retcode = subprocess.call(['rm'] + glob('*.FTZ'), stderr=subprocess.DEVNULL)
    return False
