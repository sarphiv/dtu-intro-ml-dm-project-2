#%%
import numpy as np
from models.mean_predictor import MeanPredictor
from models.linear_regression import LinearRegression

from validators.cross_validator import CrossValidator

#%%
#Define loss function
def loss_fn(pred, label):
    return np.mean((pred - label)**2)

#Create features
X = np.expand_dims(np.arange(100000), 1)

#Create labels
noise = np.random.normal(0, 2, len(X))
y = 4 * X[:, 0] + noise


#%%
#Models
models = [MeanPredictor(), LinearRegression()]

#Create cross validator
cv = CrossValidator(n_outer=100, n_inner=40, n_workers=3, 
                    verbose = True, randomize_seed = 0)

#Cross validate
result = cv.cross_validate(X, y, models, loss_fn)
