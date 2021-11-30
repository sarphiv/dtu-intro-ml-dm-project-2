from validation.validation_model import ValidationModel
import numpy as np


class NaiveClassifier(ValidationModel):
    def train_predict(self, train_features, train_labels, test_features):
        #NOTE: Bias towards choosing first max class.
        # If this turns out to be a problem, make it randomly choose between the maxes,
        # but frankly - what's the probability of this accidentally happening?...
        
        #Sum up all classes and get index of most represented class
        max_class = np.argmax(train_labels.sum(axis=0))
        #Create zero array for one-hot-encoded predictions
        y_predict = np.zeros((test_features.shape[0], train_labels.shape[1]))
        #Set most represented class as prediction
        y_predict[:, max_class] = 1

        #Return prediction
        return y_predict