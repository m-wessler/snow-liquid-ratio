
# coding: utf-8

# In[ ]:


import sys
import subprocess

import numpy as np
import pandas as pd
import xarray as xr

from glob import glob
from datetime import datetime, timedelta

import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


obdir = '/uufs/chpc.utah.edu/common/home/steenburgh-group10/mewessler/observations/'


# #### Use the automated data as a baseline/sanity check
# Data is obtained from mesowest API - May use the timeseries or precipitation service<br>
# https://developers.synopticdata.com/mesonet/v2/stations/timeseries/<br>
# https://developers.synopticdata.com/mesonet/v2/stations/precipitation/<br>
# To save long records, will need to access as csv format not json
# 

# In[ ]:


site_auto = 'CLN_AUTO'
f_auto = glob(obdir + '%s*.csv'%site_auto)[0]

# Remove the units row (SI - C, mm)
data_auto = pd.read_csv(f_auto, low_memory=False)[1:]
data_auto = data_auto.drop(columns='Station_ID')

renames_auto = {'Date_Time':'time', 
                'precip_accum_one_hour_set_1':'swe_auto_mm',
                'snow_interval_set_1':'snow_auto_mm', 
                'snow_depth_set_1':'depth_auto_mm', 
                'air_temp_set_1':'tsfc_c'}

data_auto = data_auto.rename(columns=renames_auto)

# Fix the datatypes
data_auto['time'] = data_auto['time'].astype('datetime64[ns]')
data_auto = data_auto.set_index('time')

data_auto = data_auto.astype({k:np.float32 for k in data_auto.keys()})

# Noticed later in plotting that the data is labeled mm but acutally in cm, fix here
data_auto['snow_auto_mm'] /= 10

data_auto[:10]


# #### Isolate precipitating (snow) periods

# In[ ]:


# Since the ERA 3h may be used, reduce hourly to 3h data
kwargs = {'rule':'3H', 'base':0, 
          'label':'right', 'closed':'right'}

data_auto3h = pd.DataFrame([
    data_auto['tsfc_c'].resample(**kwargs).max(),
    data_auto['swe_auto_mm'].resample(**kwargs).sum(),
    data_auto['snow_auto_mm'].resample(**kwargs).sum()]).T

data_auto3h[:10]


# In[ ]:


# Write these out to a file later
precip_periods = data_auto[(
    (data_auto['snow_auto_mm'] > 0.) & 
    (data_auto['swe_auto_mm'] > 0.) &
    (data_auto['tsfc_c'] <= 6.))].index

precip_periods3h = data_auto3h[(
    (data_auto3h['snow_auto_mm'] > 0.) & 
    (data_auto3h['swe_auto_mm'] > 0.) &
    (data_auto3h['tsfc_c'] <= 6.))].index

precip_periods[:5], precip_periods3h[:5]


# In[ ]:


site = 'CLN'
f = glob(obdir + '%s*.csv'%site)[0]

data_raw = pd.read_csv(f, low_memory=False)
data_raw = data_raw.set_index(['DATE'])

data_time = np.array([[datetime.strptime(d+t, '%m/%d/%y %H%M') for t in [' 0400', ' 1600']] for d in data_raw.index]).flatten()
data_raw = np.array([[(data_raw.loc[d]['%sWATER'%t], data_raw.loc[d]['%sSNOW'%t]) for t in ['0400', '1600']] for d in data_raw.index])
data_raw = data_raw.reshape(-1, 2)

data = pd.DataFrame([data_time, data_raw[:, 0], data_raw[:, 1]]).T
data = data.rename(columns={0:'time', 1:'swe_in', 2:'snow_in'}).set_index('time')

# Convert in to mm
data['swe_mm'] = data['swe_in'] * 25.4
data['snow_mm'] = data['snow_in'] * 25.4
data = data.drop(columns=['swe_in', 'snow_in'])

data[:10]


# #### Resample the hourlies to match the 12h

# In[ ]:


data_auto_swe12 = data_auto['swe_auto_mm'].resample('12H', closed='right', label='right', base=4).sum()
data_auto_snow12 = data_auto['snow_auto_mm'].resample('12H', closed='right', label='right', base=4).sum()
data_auto_12 = pd.DataFrame([data_auto_swe12, data_auto_snow12]).T
data_auto_12[:10]


# In[ ]:


data_all = data.merge(data_auto_12, on='time')
data_all[:10]


# #### Ensure that the data isn't mislabeled by plotting/visualizing segments
# These should be correct, but check that the beginning of period isn't labeled instead of end, etc.<br>
# A time-shift correction can be applied if needed. Keep in mind the automated data will likely <br>
# underreport vs the manual observations so consider the timing more than event size

# In[ ]:


# I think the Alta dataset is labeled with the period START not END... 
# 12 hour time shift seems to fix the issue with the data
time_shift = 12

fig, axs = plt.subplots(3, 1, figsize=(20, 12), facecolor='w')
d0 = datetime(2018, 3, 1, 0, 0)
d1 = datetime(2018, 4, 1, 0, 0)

mask = ((data_all.index > d0) & (data_all.index <= d1) & 
        (data_all['swe_mm'] > 0.) & (data_all['snow_mm'] > 0.))

axs[0].set_title('%s SWE'%site)

axs[0].plot(data_all.loc[mask].index + timedelta(hours=time_shift), data_all.loc[mask, 'swe_mm'], color='C0')
axs[0].scatter(data_all.loc[mask].index + timedelta(hours=time_shift), data_all.loc[mask, 'swe_mm'], 
               marker='*', s=150, linewidth=0.5, color='C0')

axs[0].plot(data_all.loc[mask].index, data_all.loc[mask, 'swe_auto_mm'], color='C1')
axs[0].scatter(data_all.loc[mask].index, data_all.loc[mask, 'swe_auto_mm'], 
               marker='*', s=150, linewidth=0.5, color='C1')

axs[1].set_title('%s SNOW'%site)

axs[1].plot(data_all.loc[mask].index + timedelta(hours=time_shift), data_all.loc[mask, 'snow_mm'], color='C0')
axs[1].scatter(data_all.loc[mask].index + timedelta(hours=time_shift), data_all.loc[mask, 'snow_mm'], 
               marker='*', s=150, linewidth=0.5, color='C0')

axs[1].plot(data_all.loc[mask].index, data_all.loc[mask, 'snow_auto_mm'], color='C1')
axs[1].scatter(data_all.loc[mask].index, data_all.loc[mask, 'snow_auto_mm'], 
               marker='*', s=150, linewidth=0.5, color='C1')

axs[2].set_title('%s SLR'%site)

axs[2].scatter(data_all.loc[mask].index + timedelta(hours=time_shift), data_all.loc[mask, 'snow_mm']/data_all.loc[mask, 'swe_mm'], 
               marker='*', s=150, linewidth=0.5, color='C0', label='slr')
axs[2].scatter(data_all.loc[mask].index + timedelta(hours=time_shift), data_all.loc[mask, 'snow_auto_mm']/data_all.loc[mask, 'swe_auto_mm'], 
               marker='*', s=150, linewidth=0.5, color='C1', label='slr_auto')

axs[2].set_ylim([0, 50])

for ax in axs:
    ax.set_xlim([d0, d1])
    ax.legend()
    ax.grid()

plt.close()


# #### Apply the time shift permanantly and save out (to netCDF or .pd?)

# In[ ]:


# We only want to time shift the MANUAL observations! Break out the values and shift the array as needed
data_shift = data.copy(deep=True)
data_shift.index = data_shift.index + timedelta(hours=time_shift)

data_save = data_shift.merge(data_auto_12, on='time')
data_save[:5]


# #### Now the heavy lifting: extract a profile from the ERA5 and combine with the observation data
# This will allow us to have our predictor set and target variable in one dataset to train a Machine Learning model

# #### Determine ERA5 lat/lon gridpoint
# Import the metadata file and take what's needed

# In[ ]:


meta_file = glob(obdir + '*Metadata*.xlsx')[0]
metadata = pd.read_excel(meta_file).set_index('code').loc[site]

# Determine the start, end years from data if metadata is nan
start, end = metadata['start'], metadata['end']
start = data_save.index[0].year if np.isnan(start) else start
start = data_save.index[0].year if data_save.index[0].year < start else start
end = data_save.index[-1].year if np.isnan(end) else end
end = data_save.index[-1].year if data_save.index[-1].year > end else end
metadata['start'], metadata['end'] = start, end

# Determine the lat, lon of the site from the metadata
site_lat, site_lon = metadata['lat'], metadata['lon']
site_elev = metadata['elevation_m']

metadata


# #### Check to see if a profile for this lat/lon exists

# In[ ]:


# Import the era5_orog file to check the lat/lon grid
era5_orog_file = '/uufs/chpc.utah.edu/common/home/steenburgh-group10/mewessler/era5/era5_orog.nc'
era5_orog = xr.open_dataset(era5_orog_file)['z'].isel(time=0)
era5_orog = era5_orog.rename({'latitude':'lat', 'longitude':'lon'})
era5_lat, era5_lon = era5_orog['lat'], era5_orog['lon']

# Find the index of the correct lat lon
idx1d = (np.abs(era5_lon - site_lon) + np.abs(era5_lat - site_lat))
idx = np.unravel_index(np.argmin(idx1d, axis=None), idx1d.shape)

# Subset and convert gpm to m
era5_g = 9.80665
era5_orog = era5_orog.isel(lat=idx[1], lon=idx[0])/era5_g

era5_orog


# In[ ]:


# A string based on the lat and lon of the nearest era5 gridpoint
era5_prof_file = 'era5prof_{}N_{}W.nc'.format(
    era5_orog['lat'].values, abs(era5_orog['lon'].values))

# if isfile(era5_prof_file):
    # open profile
    # check start, end 
    # if start, end outside of bounds, produce the missing years
    # call the profile script with arguments for lat, lon, start, end
    # concatenate the variables for new years, save output
    # concatenate new years with old years, save output
    
# else:
    # if profile is not found produce new
    # call the profile script with arguments for lat, lon, start, end
    # concatenate the variables, save output
    
cmd = ('source activate downscaled_slr; python ' + 
       '/uufs/chpc.utah.edu/common/home/u1070830/code/model-tools/era5/' + 
       'extract_profile.py {} {} {} {}'.format(
            era5_orog['lat'].values, era5_orog['lon'].values, start, end))

# Notebook
# p = subprocess.Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE)
# output, err = p.communicate()

# Scripted
subprocess.run(cmd, shell=True, stderr=sys.stderr, stdout=sys.stdout)


# ## Task List:
# #### Import profile data in the aggregate
# #### Export the compiled data as netCDF
# #### Now all training gridpoints/sites will be in a consistent format and easy to aggregate!
# We can either expand the level variables here and calculate derived variables now or in a later script
