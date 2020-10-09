import os
import shlex
import numpy as np
import xarray as xr
import pandas as pd

from glob import glob
from os.path import isfile
from functools import partial
from subprocess import Popen, call, PIPE
from multiprocessing import get_context

os.environ['OMP_NUM_THREADS'] = '1'
mp_method = 'spawn'

python = '/uufs/chpc.utah.edu/common/home/u1070830/anaconda3/envs/xlab/bin/python '
script_dir = '/uufs/chpc.utah.edu/common/home/u1070830/code/snow-liquid-ratio/'
era5_script_dir = '/uufs/chpc.utah.edu/common/home/u1070830/code/model-tools/era5/'
obs_path = '/uufs/chpc.utah.edu/common/home/steenburgh-group10/mewessler/observations/'
era5_path = '/uufs/chpc.utah.edu/common/home/steenburgh-group10/mewessler/era5/'
gfs_path = '/uufs/chpc.utah.edu/common/home/steenburgh-group10/mewessler/gfs/'

def generic_ingest(site):
    
    ingest_script = script_dir + 'data-tools/generic_data_ingest.py'
    
    if len(glob(obs_path + 'clean/%s_*.pd'%site)) > 0:
        print('File exists, skipping: %s'%site)
    else:
        try:
            run_cmd = (python + ingest_script + ' %s'%site)
            P = Popen(run_cmd, shell=True, stdout=PIPE, stderr=PIPE)
            output, err = P.communicate()

        except:
            print('Ingest %s failed to run'%site)

        else:            
            if len(glob(obs_path + 'clean/%s_*.pd'%site)) > 0:
                print('Ingest %s success'%site)
                
            else:
                print('Ingest %s failed to write'%site)
                
    return None

def extract_profiles(site, metadata):

    site_lat, site_lon = metadata.loc[site, ['lat', 'lon']].values
    
    # We need to first determine the output filename (All ERA5 profiles are xx.xxN, xxx.xxW)
    sample = xr.open_dataset(era5_script_dir + 'era5_sample.nc')
    a = abs(sample['latitude']-site_lat)+abs(sample['longitude']-360-site_lon)
    yi, xi = np.unravel_index(a.argmin(), a.shape)

    lat = sample.isel(latitude=yi, longitude=xi)['latitude']
    lon = sample.isel(latitude=yi, longitude=xi)['longitude'] - 360
    
    era5_prof_file = 'era5prof_%.2fN_%.2fW.nc'%(lat, abs(lon))

    print('\nSite: %s %.3f %.3f\n%s\n'%(site, site_lat, site_lon, era5_prof_file))
    
    iter_count = 0
    while not isfile(era5_path + '/profiles/' + era5_prof_file):
        iter_count += 1
        
        for run_cmd in [
            python + era5_script_dir + 'extract_profile.py %.2f %.2f 1980 2020'%(lat, lon),
            python + era5_script_dir + 'aggregate_profile.py %.2f %.2f'%(lat, lon)]:
        
            P = Popen(shlex.split(run_cmd), shell=False, stdout=PIPE)
        
            while True:
                output = P.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    print(output.strip())

    print('%s Complete'%era5_prof_file)
    return None

def pair_profiles(site):
    
    paired_files = glob(obs_path + 'combined/%s_*.pd'%site)
            
    if len(paired_files) == 0:
        
        run_cmd = python + script_dir + 'core/pair_era5profiles.py %s'%site
        print(run_cmd)
        
        P = Popen(run_cmd, shell=True, stdout=PIPE, stderr=PIPE)
        output, err = P.communicate()

        paired_files = glob(obs_path + 'combined/%s_*.pd'%site)
        
    if len(paired_files) > 0:
        print('Complete: \n\t%s'%'\n\t'.join(paired_files))
    else:
        print('\nNo ERA5 Profile Pairs for %s'%site)
    
    return None

if __name__ == '__main__':
    metadata = pd.read_excel(obs_path + 'Dataset_Metadata.xlsx')

    # Temp fix to use CLNX instead of CLN data
    metadata['code'] = metadata['code'].replace('CLN', 'CLNX')
    metadata = metadata.set_index('code')

    site_list = metadata.index.values.astype('str')
    #site_list = ['BSNFJE', 'BSNFDC', 'BSNFEX']
    print('Sites to process:', site_list)

    worker_cap = 5
    n_workers = len(site_list) if len(site_list) < worker_cap else worker_cap

    with get_context(mp_method).Pool(n_workers) as p:
        p.map(generic_ingest, site_list)
        p.close()
        p.join()

    extract_profiles_mp = partial(extract_profiles, metadata=metadata)
    [extract_profiles_mp(site) for site in site_list]
#     with get_context(mp_method).Pool(n_workers) as p:
#         p.map(extract_profiles_mp, site_list)
#         p.close()
#         p.join()
        
#     with get_context(mp_method).Pool(n_workers) as p:
#         p.map(pair_profiles, site_list)
#         p.close()
#         p.join()

    # Extract GFS profiles here!

#     print('Data Pre-Processing Completed...')
