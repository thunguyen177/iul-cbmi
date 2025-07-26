from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from load_data import load_data
from utils import introduce_missing_data, cls_metrics, reg_metrics
import numpy as np
import time
from sklearn.model_selection import train_test_split
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler

def get_acc_and_run_time_RF(X_train_missing, y_train, X_test, y_test):
    
    if data_name in ['diabetes', 'housing']:
        start = time.time()
        rg = RandomForestRegressor()
        rg.fit(X_train_missing, y_train)
        rf_running_time = time.time() - start   
        accuracy_rf, f1_rf, recall_rf = reg_metrics(rg.predict(X_test), y_test)
    else:
        start = time.time()
        clf = RandomForestClassifier()
        clf.fit(X_train_missing, y_train)
        rf_running_time = time.time() - start   
    
        accuracy_rf, f1_rf, recall_rf = cls_metrics(clf.predict(X_test), y_test)
    
    return np.array([accuracy_rf, f1_rf, recall_rf, rf_running_time])

def one_run_clf_RF(X, y, test_size = 0.4, missing_rate = 0.8, test_missing = False):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= test_size)
 
    # Introduce missing data to X_train
    X_train_missing = introduce_missing_data(X_train.copy(), missing_rate)
    
    if test_missing:
        X_test_missing = introduce_missing_data(X_test.copy(), missing_rate)
        return get_acc_and_run_time_RF(X_train_missing, y_train, X_test_missing, y_test)
    else: 
        return get_acc_and_run_time_RF(X_train_missing, y_train, X_test, y_test)


def one_run_reg_RF(X, y, test_size = 0.4, missing_rate = 0.8, test_missing = False):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= test_size)
 
    # Introduce missing data to X_train
    X_train_missing = introduce_missing_data(X_train.copy(), missing_rate)
    
    scaler = MinMaxScaler()
    scaler.fit(X_train_missing, y_train)
    X_train_missing_norm = scaler.transform(X_train_missing)
    

    y_scaler = MinMaxScaler()
    y_train_norm = y_scaler.fit_transform(y_train.reshape(-1, 1))
    y_test_norm = y_scaler.transform(y_test.reshape(-1, 1))


    if test_missing:
        X_test_missing = introduce_missing_data(X_test.copy(), missing_rate)
        X_test_norm = scaler.transform(X_test_missing) 
    else: 
        X_test_norm = scaler.transform(X_test) 

    return get_acc_and_run_time_RF(X_train_missing_norm, y_train_norm, X_test_norm, y_test_norm)

def saveresult(res, data_name, missing_rate, run, dir_result, test_missing):

    if data_name in ['diabetes', 'housing']:
        df_new = pd.DataFrame(res, columns=["RF_mae", "RF_mse", "RF_rmse",  "RF_time"])
    else:    
        df_new = pd.DataFrame(res, columns=["RF_acc", "RF_f1", "RF_recall", "RF_time"])

    df_new["MissingRate"] = missing_rate
    df_new["run"] = run

    # File path
    csv_path = f"{dir_result}/{data_name}_{test_missing}_RF.csv"

    # Check if file exists
    if os.path.exists(csv_path):
        # Load existing data and append
        df_existing = pd.read_csv(csv_path)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        # Create new
        df_combined = df_new

    # Save to CSV (overwrite with combined data)
    df_combined.to_csv(csv_path, index=False)


if __name__=='__main__':
    dir_result = '/Volumes/Macintosh HD/Projects/PAPER/10.IUL-CBMI/results/csv'

    data_names = ['iris', 'liver', 'soybean', 'parkinson', 'heart', 'glass', 'car', 'diabetes', 'housing']
    missing_range =  [0.2, 0.4, 0.6, 0.8]
    test_size = 0.4
    test_missings = [True, False]
    runs = np.arange(0,30,1)

    for data_name in data_names:
        X, y = load_data(data_name)
        for test_missing in test_missings:
            for missing_rate in missing_range:
                for run in runs:
                    if data_name in ['diabetes', 'housing']:
                        res = [one_run_reg_RF(X, y, test_size = test_size, missing_rate = missing_rate, test_missing=test_missing)]
                    else:                    
                        res = [one_run_clf_RF(X, y, test_size = test_size, missing_rate = missing_rate, test_missing=test_missing)]
                    saveresult(res, data_name, missing_rate, run, dir_result, test_missing)

    
    