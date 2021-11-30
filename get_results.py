#%%
import pickle
from tabulate import tabulate
import numpy as np
from scipy.stats import t
import matplotlib.pyplot as plt

# Un-pickling the data
base_class_results = pickle.load(open("results/classification/base.p", "rb"))
log_class_results = pickle.load(open("results/classification/logistic.p", "rb"))
nn_class_results = pickle.load(open("results/classification/nn.p", "rb"))

base_reg_results = pickle.load(open("results/regression/base.p", "rb"))
line_reg_results = pickle.load(open("results/regression/linear.p", "rb"))
nn_reg_results = pickle.load(open("results/regression/nn.p", "rb"))

line_lambda_results = pickle.load(open("results/regression/linear-1-layer.p", "rb"))
log_lambda_results = pickle.load(open("results/classification/logistic-1-layer.p", "rb"))

## Explaining the different values
# E_val for all models 100x14
# print(nn_reg_results.test_err_inner)

# E_gen for all models
# print(nn_reg_results.gen_err_inner)

# E_test for best model
# print(nn_reg_results.test_err_outer)

# E_gen for best model
# print(nn_reg_results.gen_err_estimate)

#%%
# Regression table

hiddenLayers = np.arange(2,16)
lamb_lin = np.linspace(1e-2, 90, 1000)

#Defining the column in the regression table
col1 = np.arange(1,11) # outer fold
col2 = hiddenLayers[nn_reg_results.idx_best_inner] # hidden layers of best model in outer fold
col3 = nn_reg_results.test_err_outer # test error
col4 = lamb_lin[line_reg_results.idx_best_inner] # lambda value of best model in outer fold
col5 = line_reg_results.test_err_outer # test error 
col6 = base_reg_results.test_err_outer # test error

# Combining the columns 
arr = np.array([col1,col2,col3,col4,col5,col6]).T
print("Outer, neural network,  linear regression,   baseline")
print(tabulate(arr, headers=['i', 'h', 'E_test', 'lambda','E_test','E_test'],
               tablefmt='orgtbl'))

# Creating a latex compatible string
latexString = r''
digits = 6
for obj in arr:
    for ele in obj:
        latexString = latexString + r'$' + str(ele)[:digits] + r'$&'
    latexString = latexString[:-1] + r"\\"
# print(latexString)

#%% 
# Classification table
hiddenLayers = np.arange(2,16)
lamb_log = np.linspace(1e-4, 2000, 1000)

#Defining the column in the regression table
col1 = np.arange(1,11) # outer fold
col2 = hiddenLayers[nn_class_results.idx_best_inner] # hidden layers of best model in outer fold
col3 = nn_class_results.test_err_outer # test error
col4 = lamb_log[log_class_results.idx_best_inner] # lambda value of best model in outer fold
col5 = log_class_results.test_err_outer # test error
col6 = base_class_results.test_err_outer # test error 

# Combining the column 
arr = np.array([col1,col2,col3,col4,col5,col6]).T
print("Outer, neural network,  logistic regression,   baseline")
print(tabulate(arr, headers=['i', 'h', 'E_test', 'lambda','E_test','E_test'],
               tablefmt='orgtbl'))

# Creating a latex compatible string
latexString = r''
digits = 6
for obj in arr:
    for ele in obj:
        latexString = latexString + r'$' + str(ele)[:digits] + r'$&'
    latexString = latexString[:-1] + r"\\"
print(latexString)

#%%
# Calculating confidence interval
alpha = 0.05
df = 9

# Neural network
nn_reg_mean = nn_reg_results.test_err_outer.mean()
nn_reg_std = nn_reg_results.test_err_outer.std()

nn_reg_error = t.ppf(1-alpha/2, df = df, loc = nn_reg_mean, scale = nn_reg_std) # 1-alpha/2 quantile      

# Linear regression
line_reg_mean = line_reg_results.test_err_outer.mean()
line_reg_std = line_reg_results.test_err_outer.std()

line_reg_error = t.ppf(1-alpha/2, df = df, loc = line_reg_mean, scale = line_reg_std) # 1-alpha/2 quantile

# Baseline
base_reg_mean = base_reg_results.test_err_outer.mean()
base_reg_std = base_reg_results.test_err_outer.std()

base_reg_error = t.ppf(1-alpha/2, df = df, loc = base_reg_mean, scale = base_reg_std) # 1-alpha/2 quantile

# Formatting results for plotting
gen_err_x = [1,3,5]
gen_err_y = [nn_reg_mean, line_reg_mean, base_reg_mean]
diff_errors = [nn_reg_error-nn_reg_mean, line_reg_error-line_reg_mean, base_reg_error-base_reg_mean]

# Plotting
plt.figure()
plt.errorbar(gen_err_x, gen_err_y, yerr=diff_errors, fmt = 'o', color = "k",elinewidth=3,capsize=10,capthick=2)
plt.xticks((0, 1, 3, 5,6), ('','Neural Network', 'Linear regression', 'Baseline',''))
plt.title("Generalization error estimation for regression")
plt.ylabel("Generalization error")
plt.grid() 
plt.show()

#%%
# Comparing regression models pairwise

alpha = 0.05
df = 9

# Calculating the diffences in generalization error
line_nn_reg_diff = np.abs(line_reg_results.test_err_outer
                          -nn_reg_results.test_err_outer)
line_base_reg_diff = np.abs(line_reg_results.test_err_outer
                            -base_reg_results.test_err_outer)
nn_base_reg_diff = np.abs(nn_reg_results.test_err_outer
                          -base_reg_results.test_err_outer)

# Calculating mean
line_nn_reg_mean = line_nn_reg_diff.mean()
line_base_reg_mean = line_base_reg_diff.mean()
nn_base_reg_mean = nn_base_reg_diff.mean()

# Calculating standard deviations
line_nn_reg_std = line_nn_reg_diff.std()
line_base_reg_std = line_base_reg_diff.std()
nn_base_reg_std = nn_base_reg_diff.std()

# Calculating 1-alpha/2 quantiles
line_nn_reg_error = t.ppf(1-alpha/2, df = df, loc = line_nn_reg_mean, scale = line_nn_reg_std)
line_base_reg_error = t.ppf(1-alpha/2, df = df, loc = line_base_reg_mean, scale = line_base_reg_std)
nn_base_reg_error = t.ppf(1-alpha/2, df = df, loc = nn_base_reg_mean, scale = nn_base_reg_std)

# Calculating p-values
line_nn_reg_p = 2*t.cdf(-np.abs(line_nn_reg_mean),df = df, loc = 0, scale = line_nn_reg_std)
line_base_reg_p = 2*t.cdf(-np.abs(line_base_reg_mean),df = df, loc = 0, scale = line_base_reg_std)
nn_base_reg_p = 2*t.cdf(-np.abs(nn_base_reg_mean),df = df, loc = 0, scale = nn_base_reg_std)

# Formatting results for plotting
diff_x = [1,3,5]
diff_y = [line_nn_reg_mean, line_base_reg_mean, nn_base_reg_mean]
diff_errors = [line_nn_reg_error-line_nn_reg_mean, 
               line_base_reg_error-line_base_reg_mean, 
               nn_base_reg_error-nn_base_reg_mean]

# Plotting
plt.figure()
plt.errorbar(diff_x, diff_y, yerr=diff_errors, fmt = 'o', color = "k",elinewidth=3,capsize=10,capthick=2)
plt.xticks((0, 1, 3, 5, 6), ('','Linear-NN', 'Linear-Base', 'NN-Base',''))
plt.title("Model generalization error comparison")
plt.ylabel("Difference in generalization error")
plt.grid() 
plt.show()


#%%
# Plotting Gen(Lambda)
plt.plot(list(np.linspace(1e-2, 90, 1000)),line_lambda_results.gen_err_inner.squeeze(),color='k')
plt.grid()
plt.xlabel("Regularization coefficient")
plt.ylabel("Generalization error")
plt.title("Linear regression: Gen. error and Regularization")
plt.show()
plt.plot(list(np.linspace(1e-4, 2000, 1000)),log_lambda_results.gen_err_inner.squeeze(),color='k')
plt.xlabel("Regularization coefficient")
plt.ylabel("Generalization error")
plt.title("Logistic regression: Gen. error and Regularization")
plt.grid()
plt.show()



# line_lambda_results = pickle.load(open("results/linear-1-layer.p", "rb"))

# log_lambda_results = pickle.load(open("results/logistic-1-layer.p", "rb"))