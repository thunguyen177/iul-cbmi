import numpy as np
import sklearn.neighbors._base
import sys
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..','missScalable')))
from missforest import MissForest


def cbi(X_train_missing, y_train, X_test, y_test, cat_vars,
        imputer = MissForest()):  
    '''
    cbi is a technique that predict the label on test set via imputation, without any training on X_train
    the current code is for binary classification
    the label should be in {0,1}
    '''
    
    # Stack X_train_missing and y_train into D_train
    D_train = np.hstack((X_train_missing, y_train.reshape(-1, 1)))
    
    # Stack X_test and y_test into D_test
    yh = y_test.astype(float)
    yh[1==1]=np.nan
    yh
    D_test = np.hstack((X_test, yh.reshape(-1, 1)))
    
    # Stack D_train and D_test together into D
    D= np.vstack((D_train, D_test))
    
    D_imputed = imputer.fit_transform(D, cat_vars = cat_vars)
    ypred_cbi = D_imputed[len(D_train):,-1]
    
    # return the predicted labels of testing data and the test accuracy
    return D_imputed, ypred_cbi

def cbi_train(X_train, y_train, cat_vars, imputer):  
    '''
    cbmi imputation for training data, with labels
    note that the labels can also have missing values
    the current code is for binary classification
    the label should be in {0,1}
    '''
    
    # Stack X_train_missing and y_train into D_train
    D_train = np.hstack((X_train, y_train.reshape(-1, 1)))
    
    D_train_imputed = imputer.fit_transform(D_train, cat_vars = cat_vars)
    X_train_imputed = D_train_imputed[:,:-1]
    y_train_imputed = D_train_imputed[:,-1]    
    return X_train_imputed, y_train_imputed