from validation.validation_model import ValidationModel
from nn_model import NNModel  

class NeuralNetwork(ValidationModel):
    def __init__(self, n_hidden_layers, reg_factor, shape) -> None:
        super().__init__()

        #Number of hidden layers
        self.n_hidden_layers = n_hidden_layers
        #Regularization factor (lambda)
        self.reg_factor = reg_factor
        #Shape of neural network ("Square", "Triangle")
        self.shape = shape
        


    
    
    def train_predict(self, train_features, train_labels, test_features):
        
        model = NNModel(self.n_hidden_layers, self.shape)
        
        loss_func = torch.nn.MSELoss()
        
    
    
    
    optimizer = torch.optim.SGD(nn_model_sequence.parameters(), lr = 0.001)
    
    epocs = 50
    
    for epoc in range(epocs):
        #Forward propagation
        y_predict = nn_model_sequence(x)
        
    loss = loss_func(y_predict, y)
    print(f"Epoc: {epoc}, Loss: {loss.item()}")
        
    optimizer.zero_grad()
        
    #Backward propagation
    loss.backward()
        
    optimizer.step()
    
        
        
        for i in range(1):
            

        #Set up network size and shape
        


        #Train network


        #Predict
        #Discard model