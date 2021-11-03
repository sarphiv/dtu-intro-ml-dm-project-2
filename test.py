#%%
from models.nn_valid import NeuralNetworkClass
from data.loader import getData, getRegressionLabel

data = getData(exclude=["price","apr","loanprc"])
regLabels = getRegressionLabel()
#%%
network = NeuralNetworkClass(3,0.1,"Triangle","Regression")
network.train_predict(data,regLabels,data)
