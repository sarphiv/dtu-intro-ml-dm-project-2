import torch.nn as nn
import torch.nn.functional as F


class NNModel(nn.Module):
    def __init__(self, n_nodes_per_layer, n_in, n_out, purpose) -> None:
        super(NNModel, self).__init__()
        
        self.n_in = n_in
        self.n_out = n_out
        self.purpose = purpose
        self.n_nodes_per_layer = n_nodes_per_layer

        if n_nodes_per_layer < 2:
            raise ValueError("Number of hidden layers must be above 1")



        act_inner_fn = lambda: nn.Tanh()

        layers = [nn.Linear(self.n_in, n_nodes_per_layer),
                act_inner_fn(),
                nn.Linear(n_nodes_per_layer,n_nodes_per_layer),
                act_inner_fn(),
                nn.Linear(n_nodes_per_layer,n_nodes_per_layer),
                act_inner_fn(),
                nn.Linear(n_nodes_per_layer,self.n_out)]

        # if shape == "triangle":
        #     #Add first input and hidden layer with activation function
        #     layers = [nn.Linear(self.n_in, self.n_out + self.n_hidden_layers - 1), act_inner_fn()]
        #     #Backtracking through the model, adding hidden layers from the output. Thus ensuring 
        #     #that it gets the shape desired
        #     for i in range(self.n_out + 1, self.n_out + self.n_hidden_layers - 1)[::-1]:
        #         #Creating the layers and appending them to our network as well as our activation functions
        #         layers.append(nn.Linear(i + 1, i))
        #         layers.append(act_inner_fn())
            
        #     #Add last layer without activation function
        #     layers.append(nn.Linear(self.n_out+1,self.n_out))


        # if shape == "rectangle":
        #     #Define the height/amount of neurons in the hidden layers
        #     height = self.n_out + self.n_hidden_layers
        #     #Add first input and hidden layer with activation function
        #     layers = [nn.Linear(self.n_in, height), act_inner_fn()]

        #     #Going through the network forwards this adds layers to add hidden layers with a consitent 
        #     # amount of neurons in each to give the hidden layers a rectangular like shape
        #     for i in range(n_hidden_layers - 2):
        #         layers.append(nn.Linear(height, height))
        #         layers.append(act_inner_fn())

        #     #Adding one last hidden layer with the output layer last 
        #     layers.append(nn.Linear(height, self.n_out))
            
            
        #If classifying, add Softmax to get probabilities
        if purpose == "classification":
            layers.append(nn.Softmax(dim=1))


        #Create model
        self.nn_model_sequence = nn.Sequential(*layers)


    def forward(self, x):
        return self.nn_model_sequence(x)
