import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.neighbors._base
import sys
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..','missScalable')))
from missforest import MissForest
from sklearn.preprocessing import MinMaxScaler
import time
from utils import introduce_missing_data, reg_metrics
from cbi import *

def one_run_reg(X, y, imputer, missing_rate, test_size, regressor, test_missing = False):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= test_size)
    # Introduce missing data to X_train
    X_train_missing = introduce_missing_data(X_train.copy(), missing_rate) 
    
    if test_missing:
        X_test = introduce_missing_data(X_test, missing_rate)  

    scaler = MinMaxScaler()
    scaler.fit(X_train_missing, y_train)
    X_train_missing_norm = scaler.transform(X_train_missing)
    X_test_norm = scaler.transform(X_test) 

    y_scaler = MinMaxScaler()
    y_train_norm = y_scaler.fit_transform(y_train.reshape(-1, 1))
    y_test_norm = y_scaler.transform(y_test.reshape(-1, 1))

    return get_mse_and_run_time(X_train_missing_norm, y_train_norm, X_test_norm, y_test_norm, imputer, regressor, test_missing = test_missing)    

    
def get_mse_and_run_time(X_train_missing, y_train, X_test, y_test, imputer, regressor, test_missing = False):
    # get the mse and running time when only the train set contains missing data
    start = time.time()
    D_imputed, ypred_cbi = cbi(X_train_missing, y_train, X_test, y_test, 
                                  cat_vars = None, imputer = imputer)
    rbi_running_time = time.time() - start   
    mae_rbi, mse_rbi, rmse_rbi = reg_metrics(ypred_cbi, y_test)


    
    if test_missing:
        # build a model on imputed Xtrain and fit regression model 
        start = time.time()
        imputer.fit(X_train_missing)
        X_train_imputed = imputer.transform(X_train_missing)
        reg = regressor.fit(X_train_imputed, y_train)
        X_test_imputed = imputer.transform(X_test)
        imputed_running_time = time.time() - start   
        mae_other, mse_other, rmse_other = reg_metrics(reg.predict(X_test_imputed), y_test)
        
    else:
        # build a model on imputed Xtrain and fit regression model 
        start = time.time()
        X_train_imputed = imputer.fit_transform(X_train_missing)
        reg = regressor.fit(X_train_imputed, y_train)
        imputed_running_time = time.time() - start   
        mae_other, mse_other, rmse_other = reg_metrics(reg.predict(X_test), y_test)
    

    return np.array([mae_rbi, mae_other, 
                     mse_rbi, mse_other, 
                     rmse_rbi, rmse_other,
                     rbi_running_time, imputed_running_time])

    
 
                       
