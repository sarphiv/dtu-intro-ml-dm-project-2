from validation.validation_model import ValidationModel
from nn_model import NNModel  
import torch.nn as nn
import torch
from torch import Tensor

class NeuralNetworkClass(ValidationModel):
    def __init__(self, n_hidden_layers, reg_factor, shape, purpose) -> None:
        super().__init__()

        #Number of hidden layers
        self.n_hidden_layers = n_hidden_layers
        #Regularization factor (lambda)
        self.reg_factor = reg_factor
        #Shape of neural network ("Square", "Triangle")
        self.shape = shape
        self.purpose = purpose
        
    
    
    
    def train_predict(self, train_features, train_labels, test_features):
        
        #n_hidden_layers, n_in, n_out, purpose, shape
        
        model = NNModel(self.n_hidden_layers, self.n_in, self.n_out, self.purpose, self.shape)
        
        if self.purpose == "Classification":
            loss_func = nn.CrossEntropyLoss()
        else: 
            loss_func = nn.MSELoss()
        
        optimizer = torch.optim.SGD(model.parameters(), lr = self.reg_factor)
        
        epocs = 50
        
        for epoc in range(epocs):
            #Forward propagation
            y_predict = model(train_features)
            
        loss = loss_func(y_predict, train_labels)
        print(f"Epoc: {epoc}, Loss: {loss.item()}")
            
        optimizer.zero_grad()
            
        #Backward propagation
        loss.backward()
            
        optimizer.step()
        
        #Wrapping data in a tensor 
        row = Tensor([test_features]).float()
    
        #Throwing it into our model
        yhat = model(row)

        #Transforming it into numpy arrays
        yhat = yhat.detach().numpy()
        


        #Predict
        #Discard model
        return yhat