#%%
import numpy as np
from models.mean_predictor import MeanPredictor
from models.linear_regression import LinearRegression

from validation.cross_validator import CrossValidator

#%%
#Define loss function
def loss_fn(pred, label):
    return np.mean((pred - label)**2)

#Create features
X = np.arange(100000)

#Create labels
noise = np.random.normal(0, 2, len(X))
y = 4 * X + noise


#%%
#Models
models = [MeanPredictor(), LinearRegression()]

#Create cross validator
cv = CrossValidator(n_outer=10, n_inner=10, n_workers=3, 
                    verbose = True, randomize_seed = 0)

#Cross validate
result = cv.cross_validate(X, y, models, loss_fn)
