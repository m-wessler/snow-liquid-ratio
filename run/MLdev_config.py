from datetime import datetime

## Stations to train, test, verify on
train_list = ['CLN', 'AGD', 'ALTA']
test_list = train_list
verif_list = train_list

## Current ERA5 verification includes 1980-01-01 00:00 thru 2020-05-31 23:59
train_start = datetime(1980, 1, 1, 0, 0)
train_end = datetime(2020, 5, 31, 23, 59)

## Current GFS verification includes 2015-01-15 00:00 thru 2020-05-31 23:59
verif_start = datetime(2015, 1, 15, 0, 0) 
verif_end = datetime(2020, 5, 31, 23, 59)