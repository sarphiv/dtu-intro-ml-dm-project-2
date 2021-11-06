#%%
import numpy as np
import matplotlib.pyplot as plt
import pickle
from models.mean_predictor import MeanPredictor
from models.linear_regression import LinearRegression
from models.nn_valid import NeuralNetworkClass

from validation.cross_validator import CrossValidator

from data.loader import getData, getRegressionLabel

#%%
#Define loss function
def loss_fn(pred, label):
    return np.mean((pred - label)**2)

#Create cross validator
cv = CrossValidator(n_outer=10, n_inner=10, n_workers=3, 
                    verbose = True, randomize_seed = 0)


#Define regression tester
def regression_test(models, name):
    #Create labels
    #X = getData(exclude=["price", "apr", "loanprc"])
    X = getData(exclude=[])
    y = getRegressionLabel()
    X =  (X - np.mean(X,axis=0))/(np.std(X,axis=0)+0.0001)
    #Cross validate
    result = cv.cross_validate(X, y, models, loss_fn)
    
    #Save to file
    pickle.dump(result, open(f"./results/regression/{name}.p", "wb"))
    
    #Return
    return result


#%%
#Linear testing
lambs = [1e-2, 1e-1, 1e0, 1e1, 4e1, 8e1, 1e2, 2e2, 3e2, 4e2, 8e2, 1e3, 1e4]
lin_models = [LinearRegression(l) for l in lambs]

results = regression_test(lin_models, "linear")

#%%
#Neural network testing
nn_params = [(i, 0, "Triangle", "Regression") for i in range(5,10)]
nn_models = [NeuralNetworkClass(*p) for p in nn_params]

results = regression_test(nn_models, "nn")

#%%
#data = getData(exclude=["price", "apr", "loanprc"])
data = getData(exclude=[])
labels = getRegressionLabel()

idx_label = np.argsort(labels)
data =  (data - np.mean(data,axis=0))/(np.std(data,axis=0)+0.0001)
model = nn_models[results.idx_best_inner[0]]
labels_predict = model.train_predict(data, labels, data)

plt.plot(labels[idx_label], labels_predict[idx_label],".")
plt.plot(labels[idx_label], labels[idx_label])
plt.show()