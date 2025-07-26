from load_data import load_data
from eval import one_run
from eval_regression import one_run_reg
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..','missScalable')))
from missforest import MissForest
from sklearn.ensemble import RandomForestRegressor

def saveresult(res, data_name, missing_rate, run, dir_result, test_missing):

    if data_name in ['diabetes', 'housing']:
        df_new = pd.DataFrame(res, columns=["CBMI_mae", "IClf_mae", "CBMI_mse", "IClf_mse", "CBMI_rmse", "IClf_rmse", "CBMI_time", "IClf_time"])
    else:    
        df_new = pd.DataFrame(res, columns=["CBMI_acc", "IClf_acc", "CBMI_f1", "IClf_f1", "CBMI_recall", "IClf_recall", "CBMI_time", "IClf_time"])

    df_new["MissingRate"] = missing_rate
    df_new["run"] = run

    # File path
    csv_path = f"{dir_result}/{data_name}_{test_missing}_results.csv"

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
    imputer = MissForest(criterion = ('friedman_mse', 'gini'), random_state = 0)

    data_names = ['iris', 'liver', 'soybean', 'parkinson', 'heart', 'glass', 'car', 'diabetes', 'housing']
    missing_range =  [0.2, 0.4, 0.6, 0.8]
    test_size = 0.4
    test_missings = [True, False]
    runs = np.arange(0,30,1)

    for data_name in data_names:
        if data_name in ['diabetes', 'housing']:
            regressor = RandomForestRegressor()
            X, y = load_data(data_name)
            for test_missing in test_missings:
                for missing_rate in missing_range:
                    for run in runs:
                        res = [one_run_reg(X, y, imputer=imputer, missing_rate = missing_rate, test_size = test_size, regressor=regressor, test_missing=test_missing)]
                        saveresult(res, data_name, missing_rate, run, dir_result, test_missing)
        else:
            X, y = load_data(data_name)
            for test_missing in test_missings:
                for missing_rate in missing_range:
                    for run in runs:
                        res = [one_run(X, y, imputer=imputer, test_size = test_size, missing_rate = missing_rate, test_missing=test_missing)]
                        saveresult(res, data_name, missing_rate, run, dir_result, test_missing)
            

        