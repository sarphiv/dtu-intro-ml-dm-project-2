#%%
import pickle
import numpy as np
import matplotlib.pyplot as plt
from validation.validation_result import ValidationResult

base_results = pickle.load(open("results/classification/base.p", "rb"))
log_results = pickle.load(open("results/classification/logistic.p", "rb"))
nn_results = pickle.load(open("results/classification/nn.p", "rb"))


#%%
def get_parameters(a: ValidationResult, b: ValidationResult):
    #Get true labels
    labels = a.test_labels_outer.argmax(axis=1)
    
    #Count amount of observations
    n = len(labels)
    
    #Get predictions
    preds_a = a.test_preds_outer.argmax(axis=1)
    preds_b = b.test_preds_outer.argmax(axis=1)

    #Get correct
    correct_a = labels == preds_a
    correct_b = labels == preds_b

    n_correct_a = np.sum(correct_a)
    n_correct_b = np.sum(correct_b)

    # #Get correct a but wrong b
    n_diff_correct_a = np.sum(correct_a > correct_b)
    # #Get correct b but wrong a
    n_diff_correct_b = np.sum(correct_b > correct_a)


    #Return parameters
    return (
        n, 
        n_correct_a, n_correct_b, 
        n_diff_correct_a, n_diff_correct_b
    )


def print_parameters(a_name, b_name, n, m_a, m_b, w_a, w_b):
    print(f"Parameters for (a: {a_name}), (b: {b_name})\n\
n: {n}\n\
m_a: {m_a}\n\
m_b: {m_b}\n\
a>b: {w_a}\n\
b>a: {w_b}")


#%%
print_parameters(
    "Base", "Logistic", 
    *get_parameters(base_results, log_results)
)

print_parameters(
    "Base", "Neural", 
    *get_parameters(base_results, nn_results)
)

print_parameters(
    "Logistic", "Neural", 
    *get_parameters(log_results, nn_results)
)