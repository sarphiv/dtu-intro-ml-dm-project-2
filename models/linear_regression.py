from validation.validation_model import ValidationModel
import sklearn.linear_model as lm
import numpy as np


class LinearRegression(ValidationModel):
    def train_predict(self, train_features, train_labels, test_features):
        m = lm.LinearRegression(fit_intercept=True).fit(
            np.expand_dims(train_features, 1), 
            train_labels)
        return m.predict(np.expand_dims(test_features, 1))