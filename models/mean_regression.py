from validation.validation_model import ValidationModel
import numpy as np


class MeanRegression(ValidationModel):
    def train_predict(self, train_features, train_labels, test_features):
        return np.repeat(np.mean(train_labels), len(test_features))