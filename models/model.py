from abc import ABC, abstractmethod

class Model(ABC):
    @abstractmethod
    def train_predict(self, train_features, train_labels, test_features):
        pass
