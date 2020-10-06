import sys
import numpy as np
import pandas as pd
import xarray as xr

from glob import glob
from os.path import isfile
from datetime import datetime, timedelta
from subprocess import Popen, PIPE

import seaborn as sns
import matplotlib.pyplot as plt

from ingest_config import *

site = sys.argv[1]
finfo = file_info[site]

f = glob(obdir + '%s*.csv'%site)[0]

data_raw = pd.read_csv(f, low_memory=False)

for old, new, unit in zip(finfo['header'], finfo['rename'], finfo['units']):
    if new is not None:
        new = new+'_%s'%unit if unit is not None else new
        data_raw = data_raw.rename(columns={old:new})
        
        if unit is not None:
            data_raw[new] = data_raw[new].astype(np.float32)
            
    else:
        data_raw = data_raw.drop(columns=old)
                
if finfo['tfix'] is not None:
    data_raw['datetime'] = [datetime.strptime(t, finfo['tfmt'])+timedelta(hours=finfo['tfix']) for t in data_raw['date']]
    
elif len(finfo['tfmt']) == 2:
    data_raw['time'] = data_raw['time'].astype(str)
    tfmt = ' '.join(finfo['tfmt'])
    data_raw['datetime'] = [datetime.strptime(' '.join(dt), tfmt) for dt in zip(data_raw['date'], data_raw['time'])]
    data_raw['datetime'] = [d+timedelta(minutes=60-d.minute) 
                            if d.minute > 30 else d-timedelta(minutes=d.minute) for d in data_raw['datetime']]
    
elif len(finfo['tfmt']) == 1:
    data_raw['datetime'] = [datetime.strptime(str(t), finfo['tfmt'][0]) for t in data_raw['datetime']]

# If we are going to attempt to assign tzinfo and parse ST vs DT, do it here and now
# For now this is timezone-naive
# Convert to UTC
data_raw['datetime_utc'] = data_raw['datetime'] + timedelta(hours=finfo['tzinfo'])

# Clean things up
data_raw = data_raw.set_index('datetime_utc')
data_raw = data_raw.sort_index()
data_raw = data_raw.drop(columns=[k for k in data_raw.keys() if (('date' in k)|('time' in k))])
    
in_mm = 25.4
cm_mm = 10
m_mm = 1000

for k in data_raw.keys():
    
    data_raw[k] = data_raw[k].replace(-999, np.nan)
    
    if k[-3:] == '_in':
        data_raw[k] *= in_mm
        data_raw = data_raw.rename(columns={k:k.replace('_in', '_mm')}).round(2)
    elif k[-3:] == '_cm':
        data_raw[k] *= cm_mm
        data_raw = data_raw.rename(columns={k:k.replace('_cm', '_mm')}).round(2)
    elif k[-2:] == '_m':
        data_raw[k] *= m_mm
        data_raw = data_raw.rename(columns={k:k.replace('_m', '_mm')}).round(2)
    elif k[-5:] == '_degF':
        data_raw[k] = (data_raw[k] - 32) * (5 / 9)
        data_raw = data_raw.rename(columns={k:k.replace('_degF', '_degC')}).round(2)
        
intervals = [k.split('_')[0].replace('snow', '') for k in data_raw.keys() if ('snow' in k)]
for i in intervals:
    data_raw.insert(0, 'slr'+i, np.round(data_raw['snow%s_mm'%i]/data_raw['swe%s_mm'%i], 0))
    data_raw['slr'+i][np.isinf(data_raw['slr'+i])] = np.nan

# Ensure data is typecast properly
data_raw = data_raw.fillna(np.nan)
for k in data_raw.keys():
    try:
        data_raw[k] = data_raw[k].astype(np.float32)
    except:
        pass
    
data_all = data_raw

time_shift = finfo['tshift'] if finfo['tshift'] is not None else 0

if 'precip_periods' not in data_all.keys():
    data_all['precip_periods'] = np.full(data_all.shape[0], fill_value='[]')
    
data_save = data_all

data_save_file = f.split('/')[-1].replace('.csv', '.pd')
print('Saving %s'%data_save_file)

data_save.to_pickle(obdir + 'clean/' + data_save_file)