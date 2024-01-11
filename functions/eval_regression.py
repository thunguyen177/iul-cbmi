import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.neighbors._base
import sys
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
from missingpy import MissForest
from sklearn.datasets import make_regression
import random 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import time
from functions.utils import *
from functions.rbi import *
from sklearn.preprocessing import StandardScaler

def one_run_reg(X, y, imputer, missing_rate, test_size, regressor, test_missing = False):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= test_size)
    # Introduce missing data to X_train
    X_train_missing = introduce_missing_data(X_train.copy(), missing_rate) 
    
    if test_missing:
        X_test = introduce_missing_data(X_test, missing_rate)   

    return get_mse_and_run_time(X_train_missing, X_train, y_train, X_test, y_test, imputer, regressor, test_missing = test_missing)    

    
def get_mse_and_run_time(X_train_missing, X_train, y_train, X_test, y_test, imputer, regressor, test_missing = False):
    # get the mse and running time when only the train set contains missing data
    start = time.time()
    ypred_rbi, mse_rbi = rbi(X_train_missing, X_train, y_train, X_test, y_test, imputer = imputer)
    rbi_running_time = time.time() - start   
    
    #train the model directly on missng data and classify directly on missing data
#     start = time.time()
#     reg = regressor.fit(X_train, y_train)
#     mse_direct = mean_squared_error(y_test, reg.predict(X_test))
#     direct_running_time = time.time() - start       
    
    if test_missing:
        # build a model on imputed Xtrain and fit regression model 
        start = time.time()
        imputer.fit(X_train_missing)
        X_train_imputed = imputer.transform(X_train_missing)
        reg = regressor.fit(X_train_imputed, y_train)
        X_test_imputed = imputer.transform(X_test)
        mse_imputed = mean_squared_error(y_test, reg.predict(X_test_imputed))
        imputed_running_time = time.time() - start   
        return np.array([mse_rbi, mse_imputed, rbi_running_time, imputed_running_time])
    else:
        # build a model on imputed Xtrain and fit regression model 
        start = time.time()
        X_train_imputed = imputer.fit_transform(X_train_missing)
        reg = regressor.fit(X_train_imputed, y_train)
        mse_imputed = mean_squared_error(y_test, reg.predict(X_test))
        imputed_running_time = time.time() - start   
        return np.array([mse_rbi, mse_imputed, rbi_running_time, imputed_running_time])
    
 
                       
