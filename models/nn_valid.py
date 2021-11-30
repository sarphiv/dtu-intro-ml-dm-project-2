from validation.validation_model import ValidationModel
from models.nn_model import NNModel  
import numpy as np
import torch.nn as nn
import torch
from torch import Tensor
from torch.utils.data import DataLoader , TensorDataset
from sklearn.model_selection import StratifiedKFold


class NeuralNetworkClass(ValidationModel):
    def __init__(self, n_hidden_layers, purpose, bagData = False) -> None:
        super().__init__()
        
        if purpose not in ["regression", "classification"]:
            raise ValueError("Unsupported purpose")



        #Number of hidden layers
        self.n_hidden_layers = n_hidden_layers
        #Shape of neural network ("Square", "Triangle")
        self.purpose = purpose
        self.epochs = 500
        self.bagData = bagData


    def train_predict(self, train_features, train_labels, test_features):
        classifying = self.purpose == "classification"
        
        
        
        #n_hidden_layers, n_in, n_out, purpose, shape
        feat_t = Tensor(train_features)
        lab_t = Tensor(train_labels)
        
        dataset = TensorDataset(feat_t, lab_t)
        if classifying and self.bagData: 
            data_batcher = DataLoader(
                dataset,
                batch_sampler=StratifiedBatchSampler(
                    lab_t.argmax(dim=1), 
                    batch_size=72, 
                    shuffle=True
                )
            )
        else:
            data_batcher = DataLoader(dataset, batch_size = 72, shuffle=True)
        
        
        model = NNModel(self.n_hidden_layers, 
                        feat_t.shape[1],
                        lab_t.shape[1] if classifying else 1, 
                        self.purpose)
        
        if classifying:
            loss_func = nn.CrossEntropyLoss()
        else: 
            loss_func = nn.MSELoss()


        learning_rate = 1e-3
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        
        for epoch in range(self.epochs):
            for x, y in iter(data_batcher):
                # Forward pass: compute predicted y by passing x to the model.
                y_pred = model(x)

                if classifying:
                    loss = loss_func(y_pred, y.argmax(dim=1))
                else:
                    loss = loss_func(y_pred.squeeze(), y)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()



        
        #Wrapping data in a tensor 
        feat_test= Tensor(test_features)
        
        #Throwing it into our model
        test_pred = model(feat_test)

        #Transforming it into numpy arrays
        test_pred = test_pred.detach().numpy().squeeze()


        #If classifying, one-hot-encode predictions
        if classifying:
            y_pred = np.zeros_like(test_pred)
            y_pred_idx = np.argmax(test_pred, axis=1)
            
            y_pred[np.arange(len(test_pred)), y_pred_idx] = 1
            
            test_pred = y_pred


        #Predict
        #Discard model
        return test_pred


class StratifiedBatchSampler:
    """Stratified batch sampling
    Provides equal representation of target classes in each batch
    Author: Reuben Feinman
    """
    def __init__(self, y, batch_size, shuffle=True):
        if torch.is_tensor(y):
            y = y.numpy()
        assert len(y.shape) == 1, 'label array must be 1D'
        n_batches = int(len(y) / batch_size)
        self.skf = StratifiedKFold(n_splits=n_batches, shuffle=shuffle)
        self.X = torch.randn(len(y),1).numpy()
        self.y = y
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            self.skf.random_state = torch.randint(0,int(1e8),size=()).item()
        for train_idx, test_idx in self.skf.split(self.X, self.y):
            yield test_idx

    def __len__(self):
        return len(self.y)