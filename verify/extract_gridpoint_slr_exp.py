import gc, os
import pickle
import cfgrib
import pygrib

import pandas as pd
import numpy as np
import xarray as xr

import matplotlib.pyplot as plt
import multiprocessing as mp

from glob import glob
from functools import reduce, partial
from datetime import datetime, timedelta
from sklearn.preprocessing import RobustScaler

os.environ['OMP_NUM_THREADS'] = '1'
mp_use_cores = 256
use_era_scaler = False

####### CONFIG ####### CONFIG ####### CONFIG #######

site, obs_interval = 'CLNX', 12
site_lat, site_lon = 40.5763, -111.6383

####### CONFIG ####### CONFIG ####### CONFIG #######

model = 'gfs0p25'
archive = '/uufs/chpc.utah.edu/common/home/steenburgh-group10/mewessler/archive/'
mlmodel_dir = '/uufs/chpc.utah.edu/common/home/steenburgh-group10/mewessler/output/slr_models/all_dev/'

date_fmt = '%Y%m%d'
datetime_fmt = '%Y%m%d%H'

def ingest_gfs(f, grid_index):
    
    lon_idx, lat_idx = grid_index[0][0], grid_index[1][0] 
    
    datasets = cfgrib.open_datasets(f)

    keep_keys = ['tp', 'q', 't', 'u', 'v', 'absv', 'w', 'gh', 'r', 'd', 
                  'u10', 'v10', 'u100', 'v100', 't2m', 'd2m', 
                  'cape', 'prmsl', 'sp', 'orog', 'hpbl']

    sfc, iso = [], []

    for ds in datasets:
        
        ds = ds.isel(latitude=lat_idx, longitude=lon_idx).load()

        key_match = np.array(list(ds.data_vars))[np.isin(list(ds.data_vars), keep_keys)]

        if len(key_match) > 0:

            dims = ds.dims.keys()
            coords = ds[key_match].coords

            if ('heightAboveGround' in coords) & ('heightAboveGround' not in dims):
                sfc.append(ds[key_match].drop('heightAboveGround'))

            elif 'isobaricInhPa' in coords:
                iso.append(ds[key_match])

            elif (('surface' in coords)|('meanSea' in coords)):
                sfc.append(ds[key_match])

            elif 'prmsl' in list(ds.data_vars):
                sfc.append(ds['prmsl'])

            else:
                pass

        else:
            pass

    sfc = xr.merge(sfc).drop('t')
    iso = xr.merge(iso).rename({'isobaricInhPa':'level'})
    iso = iso.sel(level=iso.level[::-1])

    sfc['longitude'] = sfc['longitude'] - 360
    iso['longitude'] = iso['longitude'] - 360
    
    return [sfc.drop(['surface', 'meanSea', 'step']), 
            iso.drop('step')]

def process_slr(valid, interval, grid_index):
    
    init = valid - timedelta(hours=interval)
        
    f0, f1, fi = 24-interval+3, 24, 3
    fhrs = ['f%03d'%i for i in np.arange(f0, f1+1, fi)]

    flist = glob(archive + init.strftime(date_fmt) + 
                 '/models/%s/*%s*.grib2'%(model, init.strftime(datetime_fmt)))[1:]

    flist = [f for f in flist if f.split('.')[-3] in fhrs]
    
    try:
        returns = [ingest_gfs(f, grid_index) for f in flist]
        returns = np.array(returns, dtype=object)
        sfc, iso = returns[:, 0], returns[:, 1]
    except:
        #print('%s v %s failed'%(init, valid))
        return None #[valid, np.nan, np.nan, np.nan]
    
    else:
        iso = xr.concat(iso, dim='valid_time').drop('time').rename({'valid_time':'time'}).sortby('time')
        sfc = xr.concat(sfc, dim='valid_time').drop('time').rename({'valid_time':'time'}).sortby('time')

        u, v = iso['u'], iso['v']
        wdir = 90 - np.degrees(np.arctan2(-v, -u))
        wdir = xr.where(wdir <= 0, wdir+360, wdir)
        wdir = xr.where(((u == 0) & (v == 0)), 0, wdir)

        iso['dir'] = wdir
        iso['spd'] = np.sqrt(u**2 + v**2)

        for hgt in [10, 100]:

            u, v = sfc['u%d'%hgt], sfc['v%d'%hgt]
            wdir = 90 - np.degrees(np.arctan2(-v, -u))
            wdir = xr.where(wdir <= 0, wdir+360, wdir)
            wdir = xr.where(((u == 0) & (v == 0)), 0, wdir)

            sfc['dir%dm'%hgt] = wdir
            sfc['spd%dm'%hgt] = np.sqrt(u**2 + v**2)

        orog = sfc.orog
        gh = iso.gh

        lowest_level = np.full(orog.shape, fill_value=np.nan)
        lowest_level_index = np.full(orog.shape, fill_value=np.nan)

        for i, level in enumerate(iso['level']):

            lev_gh = gh.sel(level=level)
            lowest_level = xr.where(orog >= lev_gh, level.values, lowest_level)
            lowest_level_index = xr.where(orog >= lev_gh, i, lowest_level_index)

        lowest_level_index = xr.where(np.isnan(lowest_level), 0, lowest_level_index)
        lowest_level = xr.where(np.isnan(lowest_level), 1000, lowest_level)

        df = []
        match_rename = {'absv':'vo', 'gh':'z', 'hpbl':'blh', 'prmsl':'msl', 'tp':'swe_mm',
                       'u10':'u10m', 'v10':'v10m', 'u100':'u100m', 'v100':'v100m'}

        # Loop over each variable in the xarray
        for ds in [iso, sfc.drop('orog')]:

            for var_name in ds.data_vars:

                new_var_name = match_rename[var_name] if var_name in match_rename.keys() else var_name
                # print('Reducing (%s) to %s index level AGL'%(var_name, new_var_name))

                var = ds[var_name]

                if 'level' in var.coords:

                    for i in np.arange(10):

                        var_agl = np.full(shape=(orog.shape), fill_value=np.nan)

                        for j, level in enumerate(iso['level']):

                            var_agl = xr.where(lowest_level_index+i == j, var.isel(level=j), var_agl)

                            # Record the levels used, should match lowest_level array, sanity check
                            # var_agl[i, :, :] = xr.where(lowest_level_index+i == j, level, var_agl[i, :, :])

                        # We could ho ahead and append to the pandas dataframe here 
                        # at the completion of each level (_01agl, _02agl...)
                        # We will have to use [(time), lat, lon] as a multiindex
                        var_agl = xr.DataArray(var_agl, 
                             dims=['time'], 
                             coords={'time':ds['time'],
                                     'latitude':ds['latitude'], 
                                     'longitude':ds['longitude']})

                        df.append(var_agl.to_dataframe(name='%s_%02dagl'%(new_var_name.upper(), i+1)))

                        del var_agl
                        gc.collect()

                else:

                    var_agl = xr.DataArray(var.values, 
                        dims=['time'], 
                        coords={'time':ds['time'],
                            'latitude':ds['latitude'], 
                             'longitude':ds['longitude']})

                    df.append(var_agl.to_dataframe(name='%s'%new_var_name.upper()))

        df = reduce(lambda left, right: pd.merge(left, right, on=['time', 'latitude', 'longitude']), df)
        df = df.rename(columns={'SWE_MM':'swe_mm'})

        scaler_file = glob(mlmodel_dir + '*scaler*')[-1]
        stats_file = glob(mlmodel_dir + '*train_stats*')[-1]
        model_file = glob(mlmodel_dir + '*SLRmodel*')[-1]

        if use_era_scaler == True:
            with open(scaler_file, 'rb') as rfp:
                scaler = pickle.load(rfp)
        else:
            scaler = RobustScaler(quantile_range=(25, 75))

        with open(stats_file, 'rb') as rfp:
            train_stats, train_stats_norm = pickle.load(rfp)
            model_keys = train_stats.keys()

        with open(model_file, 'rb') as rfp:
            SLRmodel = pickle.load(rfp)

        df = df.loc[:, model_keys]
        scaler = scaler.fit(df)

        df_norm = pd.DataFrame(scaler.transform(df), index=df.index, columns=df.keys())

        slr = pd.DataFrame(SLRmodel.predict(df_norm), 
                           index=df_norm.index, columns=['slr']
                          ).to_xarray()['slr']

        slr = xr.where(slr < 0, 0, slr)

        swe = df['swe_mm']
        snow = swe * slr

        try:
            if np.nanmax(swe) > 0:
                slr_weighted = round(snow.sum()/swe.sum(), 1)
            else:
                slr_weighted = np.nan
        except:
            slr_weighted = np.nan

        valid = pd.to_datetime(slr[-1].time.values)

        print('%s %.2f %.2f %.2f'%(valid, snow.sum(), swe.sum(), slr_weighted))

        return [valid, snow.sum(), swe.sum(), slr_weighted]
    
if __name__ == '__main__':

    gfs_sample = xr.open_dataset('./gfs_latlon_grid.nc')
    gfs_sample['longitude'] = gfs_sample['longitude'] - 360
    gfs_lat, gfs_lon = gfs_sample['latitude'], gfs_sample['longitude']

    idx1d = (np.abs(gfs_lon - site_lon) + np.abs(gfs_lat - site_lat))
    idx1d = np.where(idx1d == np.min(idx1d))

    start, end = datetime(2015, 1, 15, 0), datetime(2020, 6, 1, 0)
    valid_times = pd.date_range(start, end, freq='12H')
     
    mp_use_cores = (mp_use_cores 
                    if mp_use_cores < len(valid_times) 
                    else len(valid_times))
                                             
    with mp.get_context('fork').Pool(mp_use_cores) as p:
        
        process_slr_mp = partial(process_slr, 
                                 interval=obs_interval, 
                                 grid_index=idx1d)
        
        extract = p.map(process_slr_mp, valid_times, chunksize=1)
        p.close()
        p.join()
    
    extract = [ex for ex in extract if ex is not None]
    extract = pd.DataFrame(np.array(extract, dtype=object), 
                           columns=['valid', 'snow_mm', 'swe_mm', 'slr']
                          ).set_index('valid').sort_index()
    
    savedir = mlmodel_dir + '/GFS_output/%s/'%site
    os.makedirs(savedir, exist_ok=True)
    
    extract.to_pickle(savedir + '%s_%d.%.3fN_%.3fW.GFS_SVR_SLR.pd'%(
        site, obs_interval, site_lat, abs(site_lon)))
    
print('Complete')