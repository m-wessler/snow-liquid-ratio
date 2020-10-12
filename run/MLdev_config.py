from datetime import datetime

# Name the training group/config:
train_name = 'all_dev'

## Stations to train, test, verify on
train_list = None #['CLN', 'AGD', 'ALTA']
test_list = train_list
verif_list = train_list

## Current ERA5 verification includes 1980-01-01 00:00 thru 2020-05-31 23:59
train_start = datetime(1980, 1, 1, 0, 0)
train_end = datetime(2020, 5, 31, 23, 59)

## Current GFS verification includes 2015-01-15 00:00 thru 2020-05-31 23:59
verif_start = datetime(2015, 1, 15, 0, 0) 
verif_end = datetime(2020, 5, 31, 23, 59)

use_intervals = [12, 24]
use_var_type = ['mean']#, 'max', 'min']
min_slr, max_slr = 2.5, 50
max_T_01agl = 0 + 273.15
min_swe_mm = 2.54
use_scaler = RobustScaler(quantile_range=(25, 75))
train_size, test_size, random_state = None, 0.33, 5

svr_tune_on = 'r2'#['mse', 'mae', 'mare', 'r2']
crange = np.arange(1, 100, 1)
erange = np.arange(0.25, 5.1, .25)

mp_cores = 64
mp_method = 'fork'