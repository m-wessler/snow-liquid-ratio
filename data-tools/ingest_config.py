# some standards... 
# precip/snow/depth should be swe12/snow12/depth12 or swe24/snow24/depth24 (or other interval)
# temperature over an interval gt 1h should be noted as tmax/tavg/tmin (use t for instantaneous only)
# manual does not need an identifier, auto should be swe12_auto/snow12_auto etc
# tzinfo should be in STANDARD time, we can adjust for daylight time later if needed
# tfmt should be split according to date columns - one column, one item; two columns, two items, etc
# use tfix to set a fixed observation hour (local time HHMM)

# 'SITE':{
# 	'header':[],
# 	'rename':[],
# 	'units':[],
# 	'tzinfo':None,
# 	'tfmt':['', ''],
# 	'tfix':None,
# 	'tshift':None,
# 	'auto_site':''
# },

obdir = '/uufs/chpc.utah.edu/common/home/steenburgh-group10/mewessler/observations/'

file_info = {
    'ALTA':{
        'header':['LOCATION','DATE','PRCP','SNOW','SNWD','TMAX','TMIN','TOBS'],
        'rename':[None, 'date', 'swe24', 'snow24', 'depth24', 'tsfc_max', 'tsfc_min', 'tsfc'],
        'units':[None, None, 'in', 'in', 'in', 'degF', 'degF', 'degF'],
        'tzinfo':-7,
        'tfmt':'%m/%d/%Y',
        'tfix':18,
        'tshift':None,
        'auto_site':'CLN',
    },
    'AGD':{
        'header':['Location','DateMST','TimeMST','MaxT','MinT','12HrSnow','24HrSnow','12HrWater','24HrWater'],
        'rename':[None, 'date', 'time', 'tsfc_max', 'tsfc_min', 'snow12', 'snow24', 'swe12', 'swe24'],
        'units':[None, None, None, 'degF', 'degF', 'in', 'in', 'in', 'in'],
        'tzinfo':-7,
        'tfmt':['%m/%d/%y', '%H:%M:%S'],
        'tfix':None,
        'tshift':None,
        'auto_site':'CLN',
    },
    'BCC':{
        'header':['Location','Date LOCAL','Time LOCAL','MaxT','MinT','12HrSnow','24HrSnow','12HrWater','24HrWater'],
        'rename':[None, 'date', 'time', 'tsfc_max', 'tsfc_min', 'snow12', 'snow24', 'swe12', 'swe24'],
        'units':[None, None, None, 'degF', 'degF', 'in', 'in', 'in', 'in'],
        'tzinfo':-7,
        'tfmt':['%m/%d/%y', '%H:%M:%S'],
        'tfix':None,
        'tshift':None,
        'auto_site':'BRC'
    },
    'SLB':{
        'header':['NAME','DATE','PRCP','SNOW','SNWD','TMAX','TMIN','TOBS'],
        'rename':[None, 'date', 'swe24', 'snow24', 'depth24', 'tsfc_max', 'tsfc_min', 'tsfc'],
        'units':[None, None, 'in', 'in', 'in', 'degF', 'degF', 'degF'],
        'tzinfo':-7,
        'tfmt':'%Y-%m-%d', #1917-08-22
        'tfix':9,
        'tshift':None,
        'auto_site':'PKCU1' #or BRC
    },
    'PVC':{
        'header':['Location','Date LOCAL','Time LOCAL','MaxT','MinT','12HrSnow','24HrSnow','12HrWater','24HrWater'],
        'rename':[None, 'date', 'time', 'tsfc_max', 'tsfc_min', 'snow12', 'snow24', 'swe12', 'swe24'],
        'units':[None, None, None, 'degF', 'degF', 'in', 'in', 'in', 'in'],
        'tzinfo':-7,
        'tfmt':['%m/%d/%y', '%H:%M:%S'],
        'tfix':None,
        'tshift':None,
        'auto_site':'UTPCY', #UTLPC
    },
    'BSNFDC':{
        'header':['DATE','TIME','SITE','DEPTH_cm','SNOW_cm','SWE_mm'],
        'rename':['date', 'time', None, 'depth24', 'snow24', 'swe24'],
        'units':[None, None, None, 'cm', 'cm', 'mm'],
        'tzinfo':-7,
        'tfmt':['%m/%d/%Y', '%H:%M'],
        'tfix':None,
        'tshift':None,
        'auto_site':None
    },
    'BSNFEX':{
        'header':['DATE','TIME','SITE','DEPTH_cm','SNOW_cm','SWE_mm'],
        'rename':['date', 'time', None, 'depth24', 'snow24', 'swe24'],
        'units':[None, None, None, 'cm', 'cm', 'mm'],
        'tzinfo':-7,
        'tfmt':['%m/%d/%Y', '%H:%M'],
        'tfix':None,
        'tshift':None,
        'auto_site':None
    },
    'BSNFJE':{
        'header':['DATE','TIME','SITE','DEPTH_cm','SNOW_cm','SWE_mm'],
        'rename':['date', 'time', None, 'depth24', 'snow24', 'swe24'],
        'units':[None, None, None, 'cm', 'cm', 'mm'],
        'tzinfo':-7,
        'tfmt':['%m/%d/%Y', '%H:%M'],
        'tfix':None,
        'tshift':None,
        'auto_site':None
    },
}