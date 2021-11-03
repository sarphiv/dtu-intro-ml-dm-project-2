from validation.validation_model import ValidationModel
from sklearn.linear_model import LogisticRegression
import numpy as np


class LogisticClassifier(ValidationModel):
    
    def __init__(self, lamb):
        if lamb == 0:
            raise ValueError("Lambda must not be zero")
        
        self.lamb = lamb
    
    def train_predict(self, train_features, train_labels, test_features):
        m = LogisticRegression(C=1/self.lamb, fit_intercept=True)
        m.fit(train_features, train_labels)
        
        return m.predict(test_features)