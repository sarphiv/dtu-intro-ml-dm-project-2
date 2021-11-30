#%%
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from models.nn_valid import NeuralNetworkClass
from models.mean_regression import MeanRegression
from models.naive_classification import NaiveClassifier
from models.linear_regression import LinearRegression
from data.loader import getRegressionFeatures, getRegressionLabels, getClassificationFeatures, getClassificationLabels
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

X = getRegressionFeatures(standardize = True)
y = getRegressionLabels(standardize = True)

#Define loss function
def loss_fn(pred, label):
    #Subtract predections from labels.
    # Incorrect will result in row summing to 2
    # Correct will result in row being zero
    # Sum all rows (equivalent to summing everything), 
    # then divide by two to get amount of mispredictions.
    # Divide by total to get error rate.
    return (np.abs(pred - label).sum() / 2) / len(pred)

#%%

# for i in tqdm(range(50,55)):
    # data = getRegressionFeatures(standardize = True, pca = True, n_pc = i)
    # regLabels = getRegressionLabels(standardize=True)

    # #Define loss function
    # def loss_fn(pred, label):
    #     #Subtract predections from labels.
    #     # Incorrect will result in row summing to 2
    #     # Correct will result in row being zero
    #     # Sum all rows (equivalent to summing everything), 
    #     # then divide by two to get amount of mispredictions.
    #     # Divide by total to get error rate.
    #     return (np.abs(pred - label).sum() / 2) / len(pred)


# network = NaiveClassifier()
# network = NeuralNetworkClass(14, "regression")
network = LinearRegression(0)
    #regLabels_predict = network.train_predict(data, regLabels, data)

    # print(loss_fn(regLabels_predict, regLabels))


    # idx_label = np.argsort(regLabels)
    # plt.plot(regLabels[idx_label], regLabels_predict[idx_label], ".")
    # plt.plot(regLabels[idx_label], regLabels[idx_label])
    # plt.show()


#%%
idx_split = 1600



y_pred = network.train_predict(X[:idx_split], y[:idx_split], X[idx_split:])
# print(loss_fn(y_pred, y[idx_split:]))


idx_label = np.argsort(y[idx_split:])
y_test = y[idx_split:][idx_label]


plt.plot(y_test, y_test)
plt.plot(list(y_test) + list(y_test)[::-1], list(y_test - 0.54) + list(y_test + 0.54)[::-1])
plt.plot(y_test, y_pred[idx_label], ".")
plt.show()
