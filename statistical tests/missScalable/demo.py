import numpy as np
"""
The 'criterion' parameter of RandomForestRegressor must be a str among 
{'poisson', 'friedman_mse', 'absolute_error', 'squared_error'}. Got 'mse' instead.

criterion for MissForest should be 
'poisson', 'friedman_mse', 'absolute_error', 'squared_error'
Default: friedman_mse
"""

nan = np.nan
X = [[1, 2, nan], [3, 4, 3], [nan, 6, 5], [8, 8, 7]]
from missforest import MissForest
imputer = MissForest()
X_imputed = imputer.fit_transform(X)