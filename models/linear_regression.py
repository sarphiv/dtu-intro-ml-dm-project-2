from validation.validation_model import ValidationModel
from sklearn.linear_model import Ridge
import numpy as np


class LinearRegression(ValidationModel):
    def __init__(self, lamb):
        self.lamb = lamb
    
    def train_predict(self, train_features, train_labels, test_features):
        m = Ridge(alpha=self.lamb, fit_intercept=True)
        m.fit(train_features, train_labels)
        
        return m.predict(test_features, 1)