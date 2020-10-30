#!/uufs/chpc.utah.edu/common/home/u1070830/anaconda3/envs/xlab/bin/python

import sys, os
import shutil
import warnings

import netCDF4 as nc
import numpy as np
import xarray as xr
import pandas as pd
import multiprocessing as mp
import matplotlib.pyplot as plt

from glob import glob
from datetime import datetime
from functools import partial

warnings.filterwarnings('ignore')
os.environ['OMP_NUM_THREADS'] = '1'

model = 'gfsds'

try:
    init = datetime.strptime(sys.argv[1], '%Y%m%d%H')
except:    
    # Call @ (00)23:00, (06)05:00, (12)11:00, (18UTC)17:00 MDT (-6)
    # Call @ (00)22:00, (06)04:00, (12)10:00, (18UTC)16:00 MST (-7)
    init = datetime.utcnow()
    init_hour = [0, 6, 12, 18][int(init.hour/6)]
    init = datetime(init.year, init.month, init.day, init_hour)

archive_dir = '/uufs/chpc.utah.edu/common/home/horel-group/archive/'
output_dir = '/uufs/chpc.utah.edu/common/home/steenburgh-group10/mewessler/wpc/oper/%s/%s/'%(model, init.strftime('%Y%m%d%H'))
data_dir = archive_dir  + '%s/models/%s/%s/'%(init.strftime('%Y%m%d'), model, init.strftime('%Y%m%d%H'))
os.makedirs(output_dir, exist_ok=True)

def netcdf4_writer(ds):
    
    itime = pd.to_datetime(ds['init'].values)
    vtime = pd.to_datetime(ds['time'].values)
    ftime = ds['fhr'].values
    
    lat, lon = ds['lat'], ds['lon']
    lon2d, lat2d = np.meshgrid(lon, lat)
    accum24_sec = 60 * 60 * 24
    
    ncfilename = output_dir + 'uutah_downscaled_gfs.no_proj.%s_f%03d_v%s.nc'%(
            itime.strftime('%Y%m%d%H'), ftime, vtime.strftime('%Y%m%d%H'))
    
    if os.path.isfile(ncfilename):
        os.remove(ncfilename)
    
    with nc.Dataset(ncfilename, 'w', format='NETCDF4') as ncfile:
        
        # Global Attributes
        ncfile.Projection = "LatLon"
        ncfile.longitude_of_prime_meridian = 0.0
        ncfile.semi_major_axis = 6378137.0
        ncfile.inverse_flattening = 298.257223563
        
        ncfile.nx = str(len(lon))
        ncfile.ny = str(len(lat))

        # Lat Lon dimensions and data
        ncfile.createDimension('lon', len(lon))
        ncfile.createDimension('lat', len(lat))

        lon_nc = ncfile.createVariable('lon', 'f4', ('lat', 'lon'))
        lon_nc.long_name = 'longitude'
        lon_nc.units = 'degrees_east'
        lon_nc.standard_name = 'longitude'
        lon_nc._CoordinateAxisType = 'Lon'

        lat_nc = ncfile.createVariable('lat', 'f4', ('lat', 'lon'))
        lat_nc.long_name = 'latitude'
        lat_nc.units = 'degrees_north'
        lat_nc.standard_name = 'latitude'
        lat_nc._CoordinateAxisType = 'Lat'

        lon_nc[:] = lon2d
        lat_nc[:] = lat2d

        # Write variable data
        apcp_24_nc = ncfile.createVariable(
            'APCP_R1', 'f4', ('lat', 'lon'), 
            fill_value=-9999.0, zlib=True, complevel=9)
        
        apcp_24_nc.long_name = 'Total Precipitation'
        apcp_24_nc.level = 'R1'
        apcp_24_nc.units = 'inches'
        apcp_24_nc.init_time = itime.strftime('%Y%m%d_%H%M%S')
        apcp_24_nc.init_time_ut = str(itime.timestamp())
        apcp_24_nc.valid_time = vtime.strftime('%Y%m%d_%H%M%S')
        apcp_24_nc.valid_time_ut = str(vtime.timestamp())
        apcp_24_nc.accum_time_sec = accum24_sec

        asnow_24_nc = ncfile.createVariable(
            'ASNOW_R1', 'f4', ('lat', 'lon'), 
            fill_value=-9999.0, zlib=True, complevel=9)
        
        asnow_24_nc.long_name = 'Total Snow'
        asnow_24_nc.level = 'R1'
        asnow_24_nc.units = 'inches'
        asnow_24_nc.init_time = itime.strftime('%Y%m%d_%H%M%S')
        asnow_24_nc.init_time_ut = str(itime.timestamp())
        asnow_24_nc.valid_time = vtime.strftime('%Y%m%d_%H%M%S')
        asnow_24_nc.valid_time_ut = str(vtime.timestamp())
        asnow_24_nc.accum_time_sec = accum24_sec

        slr_24_nc = ncfile.createVariable(
            'SLR_R1', 'f4', ('lat', 'lon'), 
            fill_value=-9999.0, zlib=True, complevel=9)
        
        slr_24_nc.long_name = 'Snow Liquid Ratio'
        slr_24_nc.level = 'R1'
        slr_24_nc.units = 'ratio'
        slr_24_nc.init_time = itime.strftime('%Y%m%d_%H%M%S')
        slr_24_nc.init_time_ut = str(itime.timestamp())
        slr_24_nc.valid_time = vtime.strftime('%Y%m%d_%H%M%S')
        slr_24_nc.valid_time_ut = str(vtime.timestamp())
        slr_24_nc.accum_time_sec = accum24_sec

        mm_in = 1/25.4
        apcp_24_nc[:] = ds['apcp24'].values * mm_in
        asnow_24_nc[:] = ds['snow24'].values * mm_in
        slr_24_nc[:] = ds['slr24'].values

    print('f%03d %s written'%(ftime, os.path.basename(ncfilename)))
    
def repack_gfs_wpc(fhr, input_data, swe_threshold=0.254):
        
    i0 = np.where(input_data['fhr'] == fhr-24)[0][0]+1
    i = np.where(input_data['fhr'] == fhr)[0][0]+1
    
    select = input_data.isel(time=slice(i0, i))
        
    # Change handling of 0 SWE forecast here if need be
    mw_slr = xr.where(select['dqpf'] > swe_threshold, select['slr'], np.nan).mean(dim='time')
    mw_slr = xr.where(np.isnan(mw_slr), select['slr'].mean(dim='time'), mw_slr)
    mw_slr = mw_slr.rename('slr24')

    qpf = xr.where(select['dqpf'] > swe_threshold, select['dqpf'], np.nan).sum(dim='time')
    qpf = qpf.rename('apcp24')
    
    snow = qpf * mw_slr
    snow = snow.rename('snow24')
        
    output = xr.merge([mw_slr, qpf, snow])
    output['init'] = input_data['time'][0]
    output['fhr'] = fhr
    output['time'] = select['time'][-1]
        
    netcdf4_writer(output)
    
    return None

if __name__ == '__main__':
    
    print('Repacking WPC %s %s'%(model.upper(), init.strftime('%Y %m %d %H')))
    
    old_files = glob(output_dir + '*')
    if len(old_files) > 0:
        [os.remove(f) for f in old_files]
        
    file_list = glob(data_dir + '*.nc')

    data = xr.open_mfdataset(file_list)[['slr', 'dqpf', 'dqsf']]
    data['fhr'] = (data.time - data.time[0]).astype(int)/3.6e12

    fhrs = np.arange(24, 84+1, 12)

    repack_gfs_wpc_mp = partial(repack_gfs_wpc, input_data=data)

    with mp.get_context('fork').Pool(len(fhrs)) as p:
        p.map(repack_gfs_wpc_mp, fhrs, chunksize=1)
        p.close()
        p.join()

    tar = 'tar -czvf %suutah_downscaled_gfs.no_proj_%s.tar.gz -C %s .'%(
        '/'.join(output_dir.split('/')[:-1]), 
        pd.to_datetime(data.time.values[0]).strftime('%Y%m%d%H'), 
        output_dir)
    
    print('Compressing...')
    os.system(tar)
    
    shutil.move(
        '%suutah_downscaled_gfs.no_proj_%s.tar.gz'%(
            '/'.join(output_dir.split('/')[:-1]), 
            pd.to_datetime(data.time.values[0]).strftime('%Y%m%d%H')),
        
        '%suutah_downscaled_gfs.no_proj_%s.tar.gz'%(
            output_dir, 
            pd.to_datetime(data.time.values[0]).strftime('%Y%m%d%H')))

    print('Done...')