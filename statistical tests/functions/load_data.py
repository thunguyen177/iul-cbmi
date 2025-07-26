from sklearn.datasets import load_iris, load_diabetes, fetch_california_housing
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from ucimlrepo import fetch_ucirepo 
    
def load_data(data_name):
    if data_name == 'iris':
        iris = load_iris()
        X, y = iris.data, iris.target

    if data_name == 'liver':
        data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/liver-disorders/bupa.data',
                  sep = ",", header = None)
    
        data = data.to_numpy()
        X,y = data[:, [x for x in range(data.shape[1]) if x != 6]].astype(np.float32),data[:,-1]
        le2 = LabelEncoder()
        y = le2.fit_transform(y)

    if data_name == 'heart':
        data = pd.read_table('https://archive.ics.uci.edu/ml/machine-learning-databases/spect/SPECTF.train', header = None,sep=',')
        test = pd.read_table('https://archive.ics.uci.edu/ml/machine-learning-databases/spect/SPECTF.test',
                            header=None, sep = ',')
        data = pd.concat([data, test])
        data = data.to_numpy()
        X,y = data[:,1:], data[:,0]
        X = X.astype('float')

    if data_name == 'pakinsons':
        data = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data',
                  sep = ",")
        data = data.drop(['name'], axis = 1)
        X, y = data.drop(['status'], axis = 1), data['status']
        X = X.to_numpy()
        y = np.asarray(y)

    if data_name == 'soybean':
        data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/soybean/soybean-small.data',
                  sep = ",", header = None)

        data = data.to_numpy()
        X,y = data[:, [x for x in range(data.shape[1]) if x != 35]].astype(np.float32),data[:,-1]
        le2 = LabelEncoder()
        y = le2.fit_transform(y)

    if data_name == 'diabetes':
        diabetes = load_diabetes()
        X, y = diabetes.data, diabetes.target
    
    if data_name == 'housing':
        housing = fetch_california_housing()
        X, y = housing.data, housing.target

    if data_name == 'glass':
        # fetch dataset 
        glass_identification = fetch_ucirepo(id=42) 
        
        # data (as pandas dataframes) 
        X = glass_identification.data.features 
        y = glass_identification.data.targets 
        X = X.to_numpy()
        y = np.asarray(y)

    if data_name == 'car':
        # fetch dataset 
        car_evaluation = fetch_ucirepo(id=19) 
        
        # data (as pandas dataframes) 
        X = car_evaluation.data.features 
        y = car_evaluation.data.targets 
        import category_encoders as ce

        encoder = ce.OrdinalEncoder(cols=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])
        X = encoder.fit_transform(X)
        X = X.to_numpy().astype(float)
        
        le2 = LabelEncoder()
        y = le2.fit_transform(y)

    return X, y