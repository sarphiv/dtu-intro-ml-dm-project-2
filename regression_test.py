#%%
import numpy as np
import matplotlib.pyplot as plt
import pickle

from validation.cross_validator import CrossValidator

from models.mean_regression import MeanRegression
from models.linear_regression import LinearRegression
from models.nn_valid import NeuralNetworkClass


from data.loader import getRegressionFeatures, getRegressionLabels

#%%
#Define loss function
def loss_fn(pred, label):
    return np.mean((pred - label)**2)


#Get dataset
X = getRegressionFeatures(standardize = True)
y = getRegressionLabels(standardize = True)


#Create cross validator
cv = CrossValidator(n_outer=10, n_inner=10, n_workers=32,
                    verbose = True, randomize_seed = 9999)


#Define tester
def test(models, name):
    #Cross validate
    result = cv.cross_validate(X, y, models, loss_fn)
    
    #Save to file
    pickle.dump(result, open(f"./results/regression/{name}.p", "wb"))
    
    #Return
    return result



#%%
#Base model testing
base_models = [MeanRegression()]

results = test(base_models, "base")


#%%
#Linear model testing
lambs = list(np.linspace(1e-2, 90, 1000))
lin_models = [LinearRegression(l) for l in lambs]

results = test(lin_models, "linear")


#%%
#Neural network model testing
nn_params = [i for i in range(2, 16)]
nn_models = [NeuralNetworkClass(p, "regression") for p in nn_params]

results = test(nn_models, "nn")
