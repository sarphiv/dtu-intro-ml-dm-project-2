from validation.validation_model import ValidationModel
from sklearn.linear_model import LogisticRegression
import numpy as np


class LogisticClassifier(ValidationModel):
    def __init__(self, lamb):
        if lamb == 0:
            raise ValueError("Lambda must not be zero")
        
        self.lamb = lamb
    
    def train_predict(self, train_features, train_labels, test_features):
        #Un-one-hot-encode labels
        y_train = np.argmax(train_labels, axis=1)
        
        #Fit logistic model
        m = LogisticRegression(C=1/self.lamb, max_iter=1000)
        m.fit(train_features, y_train)
        
        #Get prediction
        y_pred = m.predict(test_features)
        
        #One-hot-encode predictions
        n_classes = train_labels.shape[1]
        n_samples = test_features.shape[0]
        
        y_one_hot = np.zeros((n_samples, n_classes))
        y_one_hot[np.arange(n_samples), y_pred] = 1


        return y_one_hot