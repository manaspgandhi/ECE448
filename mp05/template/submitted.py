import torch
import torch.nn as nn


def create_sequential_layers():
    """
    Task: Create neural net layers using nn.Sequential.

    Requirements: Return an nn.Sequential object, which contains:
        1. a linear layer (fully connected) with 2 input features and 3 output features,
        2. a sigmoid activation layer,
        3. a linear layer with 3 input features and 5 output features.
    """
    
    block = nn.Sequential(
        nn.Linear(2, 3),
        nn.Sigmoid(),
        nn.Linear(3, 5)
    )
    
    return block

def create_loss_function():
    """
    Task: Create a loss function using nn module.

    Requirements: Return a loss function from the nn module that is suitable for
    multi-class classification.
    """
    return nn.CrossEntropyLoss()

class NeuralNet(torch.nn.Module):
    def __init__(self):
        """
        Initialize your neural network here.
        """
        super().__init__()
        ################# Your Code Starts Here #################
        # I got this code from the provided website in the ipynb file
        # https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2883, 50),
            nn.GELU(),
            nn.Linear(50, 10),
            
        )
        ################## Your Code Ends here ##################

    def forward(self, x):
        """
        Perform a forward pass through your neural net.

        Parameters:
            x:      an (N, input_size) tensor, where N is arbitrary.

        Outputs:
            y:      an (N, output_size) tensor of output from the network
        """
        ################# Your Code Starts Here #################
        # I got this code from the provided website in the ipynb file
        # https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
        x = self.flatten(x)
        y = self.linear_relu_stack(x)
        return y
        ################## Your Code Ends here ##################

        
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv1d(1, 8, 4, 1, 1)
        self.pool = nn.MaxPool1d(2, 2)

        self.fc_input_size = 11528

        self.fc1 = nn.Linear(self.fc_input_size, 70)
        self.fc2 = nn.Linear(70, 70)

    def forward(self, x):
        x = x.view(-1, 1, 2883)
        x = self.pool(torch.nn.functional.relu(self.conv1(x)))

        x = x.view(-1, self.fc_input_size)
        x = self.fc2(torch.nn.functional.relu(self.fc1(x)))
        
        return x
    
def train(train_dataloader, epochs):
    """
    The autograder will call this function and compute the accuracy of the returned model.

    Parameters:
        train_dataloader:   a dataloader for the training set and labels
        test_dataloader:    a dataloader for the testing set and labels
        epochs:             the number of times to iterate over the training set

    Outputs:
        model:              trained model
    """

    ################# Your Code Starts Here #################
    """
    Implement backward propagation and gradient descent here.
    """
    
    # I got this code from the provided website in the ipynb file
    # https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
    
    # Create an instance of NeuralNet, a loss function, and an optimizer
    model = CNN()
    loss_fn = create_loss_function()
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.1, weight_decay = 0.001)
    
    
    for i in range(epochs):
        model.train()
        for batch, (X, y) in enumerate(train_dataloader):
            # Compute prediction and loss
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * 50 + len(X)

    ################## Your Code Ends here ##################

    return model
