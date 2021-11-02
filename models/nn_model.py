import torch.nn as nn
import torch.nn.functional as F


class NNModel(nn.Module):
    def __init__(self, n_hidden_layers, n_in, n_out, purpose, shape) -> None:
        super(NNModel, self).__init__()
        self.n_hidden_layers = n_hidden_layers

        self.n_in = n_in
        self.n_out = n_out
        self.purpose = purpose
        self.shape = shape

        if purpose == "Classification":
            act_func1 = lambda: nn.Sigmoid()
            act_func2 = lambda: nn.Softmax()
        else: # Regression
            act_func1 = lambda: nn.Sigmoid()
            act_func2 = lambda: nn.Sigmoid()
            
        if shape == "Triangle":
            layers = [nn.Linear(self.n_in,self.n_out+self.n_hidden_layers)]
            if(self.n_hidden_layers > 1):
                for i in range(self.n_out+1,self.n_hidden_layers+self.n_out)[::-1]:
                    nn1 = nn.Linear(i+1,i)
                    nn2 = act_func1()
                    layers.append(nn1)
                    layers.append(nn2)
                    
            layers.append(nn.Linear(self.n_out+1,self.n_out))
            # layers.append(act_func2())
        if shape == "Rectangle":
            layers = [nn.Linear(self.n_in,self.n_out+self.n_hidden_layers-1)]
            if(self.n_hidden_layers > 1):
                for i in range(self.n_hidden_layers-1):
                    nn1 = nn.Linear(self.n_out+self.n_hidden_layers-1,self.n_out+self.n_hidden_layers-1)
                    nn2 = act_func1()
                    layers.append(nn1)
                    layers.append(nn2)
            layers.append(nn.Linear(self.n_out+self.n_hidden_layers-1,self.n_out))
            layers.append(act_func2())

        self.nn_model_sequence = nn.Sequential(*layers)
    
    
    def forward(self, x):
            
        return self.nn_model_sequence(x)
        
        
        
        
        