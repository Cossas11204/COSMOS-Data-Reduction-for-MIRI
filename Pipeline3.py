# Last Modified on 2023-08-15 by Cossas

import os, glob, argparse

os.environ['CRDS_CONTEXT'] = ''
os.environ['CRDS_PATH'] = '/mnt/D/JWST_data/crds_cache/'
os.environ['CRDS_SERVER_URL'] = 'https://jwst-crds.stsci.edu'

from MIRI import run_Pipeline_3

parser = argparse.ArgumentParser()
parser.add_argument('--filter')
parser.add_argument('--instrument')
parser.add_argument('--suffix')
args = parser.parse_args()

_filter = args.filter
_instrument = args.instrument
_suffix = args.suffix

path = sorted(glob.glob(f'./{_filter}/o*/'))
for _obs_path in path:
    _obs = _obs_path.split('/')[2].strip('o')
    print(_obs)
    run_Pipeline_3(_instrument, _filter, _obs, _suffix)