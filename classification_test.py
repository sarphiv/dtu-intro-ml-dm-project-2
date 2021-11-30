#%%
import numpy as np
import matplotlib.pyplot as plt
import pickle

from validation.cross_validator import CrossValidator

from models.naive_classification import NaiveClassifier
from models.logistic_classification import LogisticClassifier
from models.nn_valid import NeuralNetworkClass

from data.loader import getClassificationFeatures, getClassificationLabels



#%%
#Define loss function
def loss_fn(pred, label):
    #Subtract predections from labels.
    # Incorrect will result in row summing to 2
    # Correct will result in row being zero
    # Sum all rows (equivalent to summing everything), 
    # then divide by two to get amount of mispredictions.
    # Divide by total to get error rate.
    return (np.abs(pred - label).sum() / 2) / len(pred)


#Get dataset
X = getClassificationFeatures(standardize = True)
y = getClassificationLabels()


#Create cross validator
cv = CrossValidator(n_outer=10, n_inner=10, n_workers=32,
                    verbose = True, randomize_seed = 9999)


#Define tester
def test(models, name):
    #Cross validate
    result = cv.cross_validate(X, y, models, loss_fn)
    
    #Save to file
    pickle.dump(result, open(f"./results/classification/{name}.p", "wb"))
    
    #Return
    return result



#%%
#Base model testing
base_models = [NaiveClassifier()]

results = test(base_models, "base")


#%%
#Logistic model testing
log_lambs = list(np.linspace(1e-4, 2000, 1000))
log_models = [LogisticClassifier(lamb) for lamb in log_lambs]

results = test(log_models, "logistic")

#%%
#Neural network model testing
nn_params = [i for i in range(2, 16)]
nn_models = [NeuralNetworkClass(p, "classification") for p in nn_params]

results = test(nn_models, "nn")
