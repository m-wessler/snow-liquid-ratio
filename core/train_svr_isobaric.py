import os, pickle
import sklearn
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import multiprocessing as mp

from glob import glob
from functools import partial

from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler

import warnings
warnings.filterwarnings('ignore')

##################
train_name = 'all'
site_list_fix = ['CLNX']#, 'BSNFJE', 'BSNFDC', 'BSNFEX'] #PVC
include_keys = ['T', 'R', 'Z', 'U', 'V']#

use_intervals = [12]#, 24]
use_var_type = ['mean']#, 'max', 'min']

min_slr, max_slr = 0, 50
max_T_650 = 0 + 273.15
min_swe_mm = 2.54

use_scaler = StandardScaler() #RobustScaler(quantile_range=(25, 75))
train_size, test_size, random_state = None, 0.33, 5

svr_tune_on = 'mae'#['mse', 'mae', 'mare', 'r2']
crange = np.arange(5, 121, 5)
erange = np.arange(0, 5.1, .25)

mp_cores = 86
mp_method = 'fork'
##################

##################
def MARE(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))

def SVR_mp(_params, train_predictors, train_target):
    
    print('.', end='')
    
    _i, _j, _C, _e = _params
    
    _model = SVR(
                C=_C, #Ridge regularization parameter for (L2)^2 penalty
                epsilon=_e, #Specifies the epsilon-tube within which no penalty is associated in the training loss function
                kernel='rbf', #'linear', 'polynomial', 'rbf'
                degree=3, #pass interger for 'polynomial' kernel, ignored otherwise
                tol=0.001, #stopping tolerance
                shrinking=False, 
                cache_size=200, 
                verbose=False)
    
    _model.fit(train_predictors, train_target)
    
    test_predictions = _model.predict(X_test_norm).flatten()
    _r2 = _model.score(X_test_norm, y_test) #sklearn.metrics.r2_score(y_test.values.flatten(), test_predictions)
    _mse = sklearn.metrics.mean_squared_error(y_test.values.flatten(), test_predictions)
    _mae = sklearn.metrics.mean_absolute_error(y_test.values.flatten(), test_predictions)
    _mare = MARE(y_test.values.flatten(), test_predictions)
    
    return (_i, _j, _C, _e, _r2, _mae, _mse, _mare, _model)


##################

if __name__ == '__main__':

    obdir = '/uufs/chpc.utah.edu/common/home/steenburgh-group10/mewessler/observations/'
    figdir = '/uufs/chpc.utah.edu/common/home/steenburgh-group10/mewessler/output/slr_figures/'
    outdir = '/uufs/chpc.utah.edu/common/home/steenburgh-group10/mewessler/output/slr_models/'

    flist = glob(obdir + 'combined/*.pd')

    # This can be a manual site list if desired
    site_list = np.unique([f.split('/')[-1].split('_')[0] for f in flist])
    
    #site_list = [s for s in site_list if 'BSNF' not in s]
    site_list = site_list_fix if site_list_fix is not None else site_list
    
    print('Training on:\n%s\n'%'\n'.join(site_list))

    # favor = 'long' #'long'

    flist = []
    for site in site_list:

        # Trim here if only using 12/24 UTC
        for interval in use_intervals:        
            flist.append(glob(obdir + 'combined/%s*%d*.pd'%(site, interval)))

    flist = np.hstack(flist)
    flist = [f for f in flist if '.isobaric.' in f]
    print('Source files:\n%s\n'%'\n'.join(flist))

    data = []
    for f in flist:

        site = f.split('/')[-1].split('_')[0]

        df = pd.read_pickle(f)
        
        keys = ['slr', 'swe_mm']
        keys.extend(np.hstack([[k for k in df.keys()
                                if ((vt in k) & (k.split('_')[0] in include_keys))] 
                               for vt in use_var_type]))
        
        df = df.loc[:, keys].rename(columns={[k for k in keys if 'swe' in k][0]:'swe_mm'})
        df = df.loc[:, :].rename(columns={[k for k in keys if 'swe' in k][0]:'swe_mm'})
        df = df.rename(columns={[k for k in keys if 'slr' in k][0]:'slr'})
        df = df.drop(columns=[k for k in keys if 'auto' in k])

        # df.insert(0, 'site', np.full(df.index.size, fill_value=site, dtype='U10'))
        doy = [int(pd.to_datetime(d).strftime('%j')) for d in df.index]
        #df.insert(2, 'day_of_year', doy)

        data.append(df.reset_index().drop(columns='time'))

    data = pd.concat(data, sort=False)

    # Treat the mean value as the instantaneous value for later applications,
    # we can change this behavior later on if desired. 
    # An alternate method would be to keep the 'mean' tag through training 
    # and choose behavior upon application
    data = data.rename(columns={k:k.replace('_mean', '') for k in data.keys()})

    data = data[data['slr'] >= min_slr]
    data = data[data['slr'] <= max_slr]
    data = data[data['T_650'] <= max_T_650]
    data = data[data['swe_mm'] >= min_swe_mm]
    
    data = data.drop(columns='swe_mm')
    
    # int(slr) for stratification (> 1 ct per class label)
    data = data.dropna()
    fac = 10
    slr = np.round(data['slr']/fac, 0)*fac
    print('\nTrain/Test Split Strata:') 
    print(slr.value_counts())

    print('\nTotal: %d'%len(data))

    # Split into train/test sets
    X_train, X_test = train_test_split(data, 
                                           test_size=test_size, train_size=train_size, 
                                           random_state=random_state, stratify=slr)

    # Perform a secondary split if separate validation set required

    # Split off the target variable now that TTsplit is done
    y_train, y_test = X_train.pop('slr'), X_test.pop('slr')
    # y_train = np.round(y_train/fac, 0)*fac

    print('\nTrain: {}\nTest: {}\nValidate: {}'.format(X_train.shape[0], X_test.shape[0], None))

    train_stats = X_train.describe()
    scaler = use_scaler.fit(X_train)

    X_train_norm = pd.DataFrame(scaler.transform(X_train.loc[:, list(X_train.keys())]), columns=X_train.keys())
    X_test_norm = pd.DataFrame(scaler.transform(X_test.loc[:, list(X_train.keys())]), columns=X_train.keys())

    print('\nNormed Sample:')
    train_stats_norm = X_train_norm.describe()
    print(train_stats_norm.T.head())

    params = {}
    params['r2'] = np.zeros((len(crange), len(erange)))
    params['mae'] = np.zeros((len(crange), len(erange)))
    params['mse'] = np.zeros((len(crange), len(erange)))
    params['mare'] = np.zeros((len(crange), len(erange)))
    params['model'] = np.empty((len(crange), len(erange)), dtype='object')
    params['epsilon'] = np.zeros((len(crange), len(erange)))
    params['C'] = np.zeros((len(crange), len(erange)))

    mp_params = np.array([[(i, j, C, e) for j, e in enumerate(erange)] 
                          for i, C in enumerate(crange)]).reshape(-1, 4)

    print('\nTrain keys:')
    print(list(X_train_norm.keys()))
    
    SVR_mp_wrapper = partial(SVR_mp, 
                             train_predictors=X_train_norm, 
                             train_target=y_train)
    
    print('\nIterations to attempt: %d'%len(mp_params))

    with mp.get_context(mp_method).Pool(mp_cores) as p:
        mp_returns = p.map(SVR_mp_wrapper, mp_params, chunksize=1)
        p.close()
        p.join()

    for item in mp_returns:

        i, j, C, e, r2, mae, mse, mare, model = item
        i, j = int(i), int(j)

        params['r2'][i, j] = r2
        params['mse'][i, j] = mse
        params['mae'][i, j] = mae
        params['mare'][i, j] = mare
        params['model'][i, j] = model
        params['epsilon'][i, j] = e
        params['C'][i, j] = C

    min_on, indexer, _ = 'R2', np.where(params['r2'] == params['r2'].max()), params['r2'].max()
    min_on, indexer, _ = 'MAE', np.where(params['mae'] == params['mae'].min()), params['mae'].min()
    min_on, indexer, _ = 'MSE', np.where(params['mse'] == params['mse'].min()), params['mse'].min()
    min_on, indexer, _ = 'MARE', np.where(params['mare'] == params['mare'].min()), params['mare'].min()

    for min_on in [svr_tune_on]:

        if min_on in ['mse', 'mae', 'mare']:
            min_max = 'Minimized'
            indexer = np.where(params[min_on] == params[min_on].min())
        elif min_on in ['r2']:
            min_max = 'Maximized'
            indexer = np.where(params[min_on] == params[min_on].max())

        r, c = indexer
        r, c = r[0], c[0]
        r, c, _

        model = params['model'][r, c]
        test_predictions = model.predict(X_test_norm)

        y_true = y_test
        y_pred = test_predictions
        print('MARE ', MARE(y_true, y_pred))

        fig, axs = plt.subplots(2, 3, facecolor='w', figsize=(24, 14))
        axs = axs.flatten()

        ax = axs[0]
        cbar = ax.pcolormesh(erange, crange, params['mae'])
        plt.colorbar(cbar, label='mae', ax=ax)
        ax.set_title('Min MAE: %.3f'%params['mae'][r, c])
        ax.scatter(params['epsilon'][r, c], params['C'][r, c], s=500, c='w', marker='+')

        ax = axs[1]
        cbar = ax.pcolormesh(erange, crange, params['mse'])
        plt.colorbar(cbar, label='mse', ax=ax)
        ax.set_title('Min MSE: %.3f'%params['mse'][r, c])
        ax.scatter(params['epsilon'][r, c], params['C'][r, c], s=500, c='w', marker='+')

        ax = axs[2]
        cbar = ax.pcolormesh(erange, crange, params['r2'])
        plt.colorbar(cbar, label='r2', ax=ax)
        ax.set_title('Max R^2: %.3f'%params['r2'][r, c])
        ax.scatter(params['epsilon'][r, c], params['C'][r, c], s=500, c='k', marker='+')

        for ax in axs[:3]:
            ax.set_xlabel('epsilon')
            ax.set_ylabel('C_val')
            ax.set_ylim([crange.min(), crange.max()])
            ax.set_xlim([erange.min(), erange.max()])

        ax = axs[3]
        maxslr = y_test.max() if y_test.max() > y_train.max() else y_train.max()

        ax.hist(y_train, bins=np.arange(0, maxslr, 2), color='g', edgecolor='k', alpha=1.0, label='Train SLR\nn=%d'%len(y_train))
        ax.hist(y_test, bins=np.arange(0, maxslr, 2), color='C0', edgecolor='k', alpha=1.0, label='Test SLR\nn=%d'%len(y_test))
        ax.legend()

        ax.set_xticks(np.arange(0, maxslr+1, 5))
        ax.set_xticklabels(np.arange(0, maxslr+1, 5).astype(int))
        ax.grid()

        ax = axs[4]
        maxslr = test_predictions.max() if test_predictions.max() > y_test.max() else y_test.max()
        maxslr += 5
        ax.scatter(y_test, test_predictions, c='k', s=50, marker='+', linewidth=0.75)
        ax.set_xlabel('Observed SLR')
        ax.set_ylabel('Predicted SLR')
        ax.plot([0, maxslr], [0, maxslr])
        ax.set_xlim([0, maxslr])
        ax.set_ylim([0, maxslr])
        ax.set_aspect('equal')
        ax.grid()

        ax = axs[5]
        error = test_predictions - y_test
        ax.hist(error, bins=np.arange(-30, 30, 2), edgecolor='k')
        ax.set_xlabel('Prediction Error')
        ax.set_ylabel('Count')
        ax.grid()

        plt.suptitle('Support Vector Regression Model\n%s\n%s on: %s\n\nepsilon %.3f\nc_val: %.3f'%(
            site_list, min_max, min_on.upper(), params['epsilon'][r, c], params['C'][r, c]))

    figpath = figdir + 'train/%s/'%train_name
    os.makedirs(figpath, exist_ok=True)
    
    numfigs = len(glob(figpath + '*_gridsearch*.png'))
    figfile = figpath + '%s_gridsearch.%02d.png'%(train_name, numfigs)

    plt.savefig(figfile)
    print('Saved: ', figfile)
    
    outpath = outdir + '%s'%train_name
    os.makedirs(outpath, exist_ok=True)
    
    for obj, obj_name in zip(
        [scaler, [train_stats, train_stats_norm], model], 
        ['scaler', 'train_stats', 'SLRmodel']):
        
        with open(outpath + '/%s_%s.%02d.pickle'%(
            train_name, obj_name, numfigs), 'wb') as wfp:
            
            pickle.dump(obj, wfp)