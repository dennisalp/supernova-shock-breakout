'''
2024-08-24, Dennis Alp, me@dennisalp.com

Search for single-frame transients in the XMM archives.
'''

from pdb import set_trace as st
import os
import sys
import subprocess
import threading
from queue import Queue

from data import read_log, download_data
from observation import Observation



def download_thread(cwd, oi, result_queue):
    print('Downloading {0:10s}'.format(oi))
    result = download_data(cwd, oi)
    print('Download of {0:10s} completed'.format(oi))
    result_queue.put((oi, result))


def main():

    if os.path.expanduser("~") == '/Users/silver':
        local = '/Users/silver/xmm/'
        drpbx = '/Users/silver/box/xmm/'
    elif os.path.expanduser("~") == '/home/dennisalp':
        local = '/media/dennisalp/dat/xmm/all/'
        drpbx = '/home/dennisalp/box/xmm/'
    else:
        print('ERROR: Unknown user. Unable to set paths')
        sys.exit(1)

    figp = local + sys.argv[1] + f'/{__file__.split('/')[-2]}/'
    cwd = local + sys.argv[1] + '/dat/'
    retcode = subprocess.call(['mkdir', '-p', cwd])
    retcode = subprocess.call(['mkdir', '-p', figp])
    os.chdir(cwd)

    obs, target, epoch = read_log(drpbx + 'observations.txt')
    result_queue = Queue()

    if len(sys.argv) > 2:
        i0 = int(sys.argv[2])
        i1 = int(sys.argv[3])
        obs = obs[i0:i1+1]
    else:
        i0 = 0

    threading.Thread(target=download_thread, args=(cwd, obs[0], result_queue)).start()

    for ii, oi in enumerate(obs):
        # Wait for the current download to complete
        current_oi, download_success = result_queue.get()

        # Start downloading the next observation if available
        if ii + 1 < len(obs):
            next_oi = obs[ii + 1]
            threading.Thread(target=download_thread, args=(cwd, next_oi, result_queue)).start()

        ii = ii + i0
        print(f'\nProcessing Obs. ID: {oi:10s} (Index: {ii:d}, Target: {target[ii]})')

        if download_success:
            observation = Observation(ii, current_oi, target[ii], epoch[ii], figp)
            observation.find_sources()
            observation.gen_products(figp)
        else:
            print(f'WARNING No data found for Obs. ID: {current_oi} (could be an RGS/OM only observation)')

if __name__ == '__main__':
    main()
