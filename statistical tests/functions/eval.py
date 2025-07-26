import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import time
from utils import introduce_missing_data, simulate_classification_data, cls_metrics
from cbi import cbi
import sklearn.neighbors._base
import sys
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..','missScalable')))
from missforest import MissForest

def one_run_simulation(imputer = MissForest(), classifier = RandomForestClassifier(),
                       num_samples = 500, num_features = 20, test_size = 0.2, missing_rate = 0.8, random_seed = 42):

    # Simulate classification data and split into train and test sets
    X_train, X_test, y_train, y_test = simulate_classification_data(num_samples, num_features, test_size, random_seed)

    # Introduce missing data to X_train
    X_train_missing = introduce_missing_data(X_train.copy(), missing_rate)
    
    return get_acc_and_run_time(X_train_missing, X_train, y_train, X_test, y_test, 
                         imputer = imputer, classifier = classifier)

def get_acc_and_run_time(X_train_missing, X_train, y_train, X_test, y_test, cat_vars = None,
                         imputer = MissForest(random_state = 0), classifier = RandomForestClassifier(), test_missing = False):
    
    # cat_vars contains the INDICES of categorical features 
    start = time.time()
    try:
        if (cat_vars.any() != None):
            cat_vars_cbi = np.hstack((cat_vars, X_train.shape[1]))  
    except:
        cat_vars_cbi = X_train.shape[1] 
    D_imputed, ypred_cbi = cbi(X_train_missing, y_train, X_test, y_test, 
                                  cat_vars = cat_vars_cbi, imputer = imputer)
    cbi_running_time = time.time() - start   
    
    accuracy_cbi, f1_cbi, recall_cbi = cls_metrics(ypred_cbi, y_test)
    
    # build a model on imputed Xtrain and classify 
    start = time.time()
    X_train_imputed = imputer.fit_transform(X_train_missing, cat_vars = cat_vars)
    classifier.fit(X_train_imputed, y_train)

    if test_missing:
        X_test_imputed = imputer.fit_transform(np.vstack((X_train_missing, X_test)),
                                               cat_vars = cat_vars)[len(X_train_missing):,:]
        #accuracy_imputed = accuracy_score(y_test, classifier.predict(X_test_imputed))
        accuracy_other, f1_other, recall_other = cls_metrics(classifier.predict(X_test_imputed), y_test)
    else:    
        #accuracy_imputed = accuracy_score(y_test, classifier.predict(X_test))    
        accuracy_other, f1_other, recall_other = cls_metrics(classifier.predict(X_test), y_test)    
    
    imputed_running_time = time.time() - start   
    
    return np.array([accuracy_cbi, accuracy_other, 
                     f1_cbi, f1_other, 
                     recall_other, recall_cbi,
                     cbi_running_time, imputed_running_time])

def one_run(X, y, imputer = MissForest(), classifier = RandomForestClassifier(),
            test_size = 0.4, missing_rate = 0.8, test_missing = False, cat_vars = None):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= test_size)
 
    # Introduce missing data to X_train
    X_train_missing = introduce_missing_data(X_train.copy(), missing_rate)
    
    if test_missing:
        X_test_missing = introduce_missing_data(X_test.copy(), missing_rate)
        return get_acc_and_run_time(X_train_missing, X_train, y_train, X_test_missing, y_test, cat_vars = cat_vars,
                         imputer = imputer, classifier = classifier, test_missing = True)
    else: 
        return get_acc_and_run_time(X_train_missing, X_train, y_train, X_test, y_test, cat_vars = cat_vars,
                         imputer = imputer, classifier = classifier)
