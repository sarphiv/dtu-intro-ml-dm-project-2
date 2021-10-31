import torch.nn as nn
import torch.nn.functional as F


class NNModel(nn.Module):
    def __init__(self, n_hidden_layers, n_in, n_out) -> None:
        super(NNModel, self).__init__()
        self.n_hidden_layers = n_hidden_layers
        #self.shape = shape
        self.n_in = n_in
        self.n_out = n_out
    
        act_func1 = lambda: nn.Tanh()
        act_func2 = lambda: nn.Sigmoid()

        
        
        self.nn_model_sequence = nn.Sequential(nn.Linear(n_in, n_hidden_layers), 
        act_func1(), 
        nn.Linear(n_hidden_layers, n_out), 
        act_func2())  
    
    #nn.Sequential([
        #input (in, out0)
        #activation
        #hidden (in1, out1)
        #action
        #hidden (in2, out2)
    # ])
    
    
    #x1, x2, x3 -> linear(3, 30) -> activation(30) -> linear(30, n)
    
    
    
    #def create_hidden_layers(self, n_hiddenlayers, shape, input_size):
    #    if shape == "Triangle":
    #        for i in range(n_hiddenlayers):
                
            #nn.Linear
    #    else:
    #        for i in range(n_hiddenlayers):
                
            #nn.Linear 
     
     
    
    
    def forward(self, x):
            
            return self.nn_model_sequence(x)
        
        
        
        
        