#%%
import numpy as np
from models.baseline import Baseline
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
models = [Baseline(), LinearRegression()]

#Create cross validator
cv = CrossValidator(n_outer=100, n_inner=40, n_workers=3, 
                    verbose = True, randomize_seed = 0)

#Cross validate
(all_losses, gen_err_inner, idx_best, 
 loss_best_outer, gen_err_outer) = cv.cross_validate(X, y, models, loss_fn)
