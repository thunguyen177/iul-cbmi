import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import random 
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error


def simulate_classification_data(num_samples=1000, num_features=10, test_size=0.2, random_seed=42):
    # Simulate classification data
    X, y = make_classification(n_samples=num_samples, n_features=num_features, 
                             n_informative=2, n_classes=2)    

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)

    return X_train, X_test, y_train, y_test

def introduce_missing_data(data, missing_rate=0.3):
    # this function generate NaN on a fully observed dataset or a dataset that has missing values
    # the missing_rate is the ratio between the number of missing entries to be generated compared to the number of observed entries
    Xshape = data.shape
    n_missing=np.count_nonzero(np.isnan(data))
    data_nan = data.flatten()
    data_nan_id = list(range(data.size)) #  all index of elements after flatten data
    nan_input_id =  np.where((np.isnan(data_nan.flatten().tolist() )))[0].tolist() # Find indices of NaN value (input data contains NaN)
  
    choice_list = [e for e in data_nan_id if e not in nan_input_id] # Just choose not NaN elements for generating NaN
    na_id = random.sample(choice_list,round(missing_rate*(data.size - n_missing))) 
    data_nan[na_id] = np.nan 
    return data_nan.reshape(Xshape)

def get_mean_std(res):
    return np.array([np.mean(res, axis = 0), np.std(res, axis = 0)]).round(3)

def cls_metrics(y_pred, y_test):
    acc = accuracy_score(y_pred, y_test)
    f1 = f1_score(y_pred, y_test, average='weighted')
    recall = recall_score(y_pred, y_test, average='weighted')
    
    return acc, f1, recall

def reg_metrics(y_pred, y_test):
    mae = mean_absolute_error(y_pred, y_test)
    mse = mean_squared_error(y_pred, y_test)
    rmse = root_mean_squared_error(y_pred, y_test)
    
    return mae, mse, rmse