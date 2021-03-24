"""
Train and Test ML Model for the Nankai Data. Outputs predictions and true solution as .txt files
Authors: Chris Liu
"""

import numpy as np
import os
import pandas as pd

from tsfresh import extract_features, select_features
from tsfresh.feature_selection import relevance
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import ComprehensiveFCParameters, settings

from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import explained_variance_score

# Find max etas
def max_eta_npy(data,gaugeno,runs):
    """
    Finds the max eta for a specific gauge and run(s)
    Input:
        data - Dictionary of timeseries data with the form: data[run number, gaugeno, :]
        gaugeno - Integer gauge number
        runs - List or range of run numbers to find the max value for
    
    Output:
        eta_max - np array containing the maximum values.
    """
    eta_max = []
    for rnum in runs:
        eta_max.append(np.amax(data[rnum,gaugeno,:]))
    return np.array(eta_max)


def train_test_split(data, target, train_runs, test_runs):
    """
    Splits data into training and testing sets given test and train indices
    
    Input:
        data -  Time series data as a npy array with format [run, eta]
        target - Array of target values for regression
        train_runs - Integer indices for the train run
        test_runs - Integer indices for the test run
        
    Output:
        train - np array containing the training set
        test - np array containing the test set
        train_target - np array of target values for train set
        test_target - np array of target values for test set
    """
    train = data[train_runs,:]
    train_target = target[train_runs]
    test = data[test_runs,:]
    test_target = target[test_runs]
    return train, test, train_target, test_target


def train_test(data, target, train_ind, test_ind, scale, model, *returns):
    """
    Trains and tests the model using specified model. 
    
    Input:
        feat - List of feature dataframes or npy array
        target - Array of target values for regression
        train_ind - Indices of the runs in the training set
        test_ind - Indices of the runs in the test set
        scale - Boolean value to denote whether features should be scaled or not to unit variance and 0 mean.
        model - sklearn model used eg. SVR, RFR
        *returns - Boolean value to determine whether scalers and models need to be returned.
        
    Output:
        pred - List of arrays of predictions from testing the model after it is trained
        target - List of arrays of targets that correspond to runs in the test set
        evs - List of explained variance scores for each dataset.
        scalers - List of standard scalers used (Optional)
        models - List of models used (Optional)
    """
    data_tmp = data
    pred = []
    tr_pred = []
    targets = []
    acc = []
    
    if returns:
        scalers = [] # empty is scale = false
        models = []

    for i in range(len(data)):
        
        # Check file format
        if isinstance(data_tmp[i],pd.DataFrame):
            train_set, test_set, train_target, test_target, = \
                train_test_split(data_tmp[i].to_numpy(), target, train_ind, test_ind)
        else:
            train_set, test_set, train_target, test_target, = \
                train_test_split(data_tmp[i], target, train_ind, test_ind)            
        
        if scale:
            scaler = StandardScaler()
            train_set = scaler.fit_transform(train_set)
            test_set = scaler.transform(test_set)
        
        model_tmp = clone(model)
        
        model_tmp.fit(train_set, train_target, sample_weight=None)
        
        pred.append(model_tmp.predict(test_set))
        tr_pred.append(model_tmp.predict(train_set))
        
        acc.append(explained_variance_score(test_target,pred[i]))
            
        targets.append(test_target)
        
        if returns:
            models.append(model_tmp)
            if scale:
                scalers.append(scaler)
            
    if returns:
        return pred, tr_pred, targets, acc, scalers, models
    else:
        return pred, tr_pred, targets, acc

if __name__ == "__main__":
    # Load data

    # for reference, change when needed
    points = ['2_05',
             '2_06',
             '2_07',
             '2_13',
             '2_14',
             '2_15',
             '2_17',
             '2_18',
             '3_23',
             '3_74',
             '5_29',
             '5_38']
    
    obs_gauges = points[:-2]

    # Directory containing the time series data. Format is [run, gauge, :]
    npydir = r'C:\Users\Chris\Desktop\Tsunami Personal\TohokuU_Data\npy'
    
    # Directory containing train test indices
    indexdir = r'C:\Users\Chris\Documents\Tsunami\jdf-autoencoder\python\data'
    
    # Directory for the predictions
    outdir = r'C:\Users\Chris\Dropbox\Tsunami Research\output\nankai'
    
    # eta_all is raw data, _s refers to smoothened data. need to manually change the code to switch datasets
    eta_all = np.load(os.path.join(npydir,'nankai_eta.npy'))
    t_all = np.load(os.path.join(npydir,'nankai_time.npy'))
#     eta_all_s = np.load(os.path.join(npydir,'nankai_eta_s.npy'))
#     t_all_s = np.load(os.path.join(npydir,'nankai_time_s.npy'))

    # Train and Test Indices
    index_train = np.loadtxt(os.path.join(indexdir, 'nankai_train_index.txt')).astype(int)
    index_test = np.loadtxt(os.path.join(indexdir, 'nankai_test_index.txt')).astype(int)


    # Featurize

    # Formating the data as input for tsfresh
    run_id = []
    g = []
    times = []
    kind = []
    twindows = [36,72] # size of input windows, multiply by 5 to get seconds
    
    runtotal = 1564
    multi_runs_used = np.arange(runtotal) # Do not exclude any runs
    multi_tstart = np.zeros(runtotal).astype(int) #do not exclude any runs

    for window in twindows:
        run_id_tmp = []
        g_tmp = []
        times_tmp = []
        kind_tmp = []

        for i in range(len(multi_runs_used)):
            for n, gauge in enumerate(obs_gauges):
                run = multi_runs_used[i]
                tstart = multi_tstart[i]
                g_data = eta_all[run, n, :]
                t_data = t_all[run, n, :]

                g_tmp.extend(g_data[tstart:tstart+window].tolist())
                times_tmp.extend(t_data[tstart:tstart+window].tolist())
                run_id_tmp.extend((np.ones(window)*run).tolist())
                kind_tmp.extend((np.ones(window)*n).tolist())

        run_id.append(run_id_tmp)
        g.append(g_tmp)
        times.append(times_tmp)
        kind.append(kind_tmp)

    # Use tsfresh
    feat_multi = []
    for i in range(len(twindows)):  
        dict = {'id':run_id[i], 'kind':kind[i], 'time':times[i], 'eta': g[i]}

        feat_tmp = extract_features(pd.DataFrame(dict), column_id='id', column_sort='time', column_kind='kind', column_value = 'eta',
                            default_fc_parameters=ComprehensiveFCParameters(), impute_function=impute)

        # drop constant features
        feat_tmp = feat_tmp.loc[:, feat_tmp.apply(pd.Series.nunique) != 1]

        feat_multi.append(feat_tmp)
    
    # create model targets
    max_5_29 = max_eta_npy(eta_all,-2,multi_runs_used)
    max_5_38 = max_eta_npy(eta_all,-1,multi_runs_used)
    
    # define model, change cache size as needed. 
    rmodel = GridSearchCV(SVR(kernel='rbf', gamma='scale', cache_size=1000),\
                          param_grid={"C": [1e-2,5e-1,1e-1,1e0, 1e1, 5e1, 1e2],\
                                      "gamma": np.logspace(-5, 0, 21)})
    
    # Train/test models
    pred_5_29, pred_tr_5_29, target_5_29, evs_5_29, scalers_5_29, models_5_29 = train_test(\
        feat_multi, max_5_29, index_train, index_test,True, rmodel,True)

    pred_5_38, pred_tr_5_38, target_5_38, evs_5_38, scalers_5_38, models_5_38 = train_test(\
        feat_multi, max_5_38, index_train, index_test ,True, rmodel,True)
    
    # Save results
    modelname = 'SVR'
    sizetst = len(index_test)
    sizetr =  len(index_train)
    
    # save the true results
    # filepaths are in Windows format
    np.savetxt(os.path.join(outdir, 'etamax_obs.txt'),\
               np.hstack((max_5_29.reshape((sizetst+sizetr,1)), max_5_38.reshape((sizetst+sizetr,1)))))

    for i in range(len(twindows)): 
        winsize = int(twindows[i]*5/60)
        etamax_pred_tst = np.hstack((pred_5_29[i].reshape((sizetst,1)),pred_5_38[i].reshape((sizetst,1))))
        etamax_pred_tr = np.hstack((pred_tr_5_29[i].reshape((sizetr,1)),pred_tr_5_38[i].reshape((sizetr,1))))
        
        np.savetxt(os.path.join(outdir, 'etamax_%s_predict_%sm.txt' %  (modelname, str(winsize))),  etamax_pred_tst)
        np.savetxt(os.path.join(outdir, 'etamax_%s_predict_tr_%sm.txt' %  (modelname, str(winsize))),  etamax_pred_tr)
                   