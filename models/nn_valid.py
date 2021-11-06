from validation.validation_model import ValidationModel
from models.nn_model import NNModel  
import torch.nn as nn
import torch
from torch import Tensor
from torch.utils.data import DataLoader , TensorDataset

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
        
    
    
    
    def train_predict(self, train_features, train_labels, test_features, epochs):
        
        #n_hidden_layers, n_in, n_out, purpose, shape
        feat_t = Tensor(train_features)
        lab_t = Tensor(train_labels)
        self.epochs = epochs
        
        dataset = TensorDataset(feat_t, lab_t)
        data_batcher = DataLoader(dataset , batch_size = 72, shuffle=True)
        
        model = NNModel(self.n_hidden_layers,
                        feat_t.shape[1], 
                        lab_t.shape[1] if self.purpose == "Classification" else 1, 
                        self.purpose, self.shape)
        
        if self.purpose == "Classification":
            loss_func = nn.CrossEntropyLoss()
        else: 
            loss_func = nn.MSELoss()
        
        
        # Use the optim package to define an Optimizer that will update the weights of
        # the model for us. Here we will use RMSprop; the optim package contains many other
        # optimization algorithms. The first argument to the RMSprop constructor tells the
        # optimizer which Tensors it should update.
        learning_rate = 1e-3
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        
        # Implement data loader
        # Remember drop last
        for epoch in range(epochs):
            for x, y in iter(data_batcher):
                # Forward pass: compute predicted y by passing x to the model.
                y_pred = model(x)

                # Compute and print loss.
                loss = loss_func(y_pred.squeeze(), y)
                

                # Before the backward pass, use the optimizer object to zero all of the
                # gradients for the variables it will update (which are the learnable
                # weights of the model). This is because by default, gradients are
                # accumulated in buffers( i.e, not overwritten) whenever .backward()
                # is called. Checkout docs of torch.autograd.backward for more details.
                optimizer.zero_grad()

                # Backward pass: compute gradient of the loss with respect to model
                # parameters
                loss.backward()

                # Calling the step function on an Optimizer makes an update to its
                # parameters
                optimizer.step()
    
            if epoch % 10 == 0:
                print(epoch, loss.item())

        print("You're doing great!")
            
            
            #Forward propagation
            #print(epoc)
            #y_predict = model(train_feat_t)
        
            #loss = loss_func(y_predict.squeeze(), Tensor(train_labels))

            #Backward propagation
            #optimizer.zero_grad()
            #loss.backward()
            #optimizer.step()
        
        #print(f"Epoc: {epoc}, Loss: {loss.item()}")
            
    


        
        #Wrapping data in a tensor 
        row = Tensor(test_features)
    
        #Throwing it into our model
        yhat = model(row)

        #Transforming it into numpy arrays
        yhat = yhat.detach().numpy().squeeze()
        


        #Predict
        #Discard model
        return yhat
