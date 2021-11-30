#%%
from sklearn.metrics.pairwise import kernel_metrics
from models.linear_regression import LinearRegression
from data.loader import getRegressionFeatures, getRegressionLabels, getClassificationFeatures, getClassificationLabels
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import statsmodels.api as sm

X, headers = getRegressionFeatures(standardize = True, return_headers = True)
y = getRegressionLabels(standardize = True)

best_lambda = 34.690

#%%
feat_idx = np.array([4, 6, 14, 22, 23, 25, 27, 28, 53])
# feat_idx = np.array([14])
X_sub = X[:, feat_idx]


print(headers[feat_idx])


#%%
model = sm.OLS(y, X_sub, hasconst=True)
results_fu = model.fit()
#Alpha (lambda) needs to be that weird expression else the cost function
# of this vs. sklearn's implementation are not the same.
results_fr = model.fit_regularized(L1_wt=0, alpha=best_lambda*2/len(y), start_params=results_fu.params,)
final = sm.regression.linear_model.OLSResults(model, 
                                              results_fr.params, 
                                              model.normalized_cov_params)

summary = final.summary()
print(summary)


#%%
network = LinearRegression(best_lambda)

print(r2_score(y, network.train_predict(X_sub, y, X_sub)))


#%%
network = LinearRegression(best_lambda)

print(r2_score(y, network.train_predict(X, y, X)))
