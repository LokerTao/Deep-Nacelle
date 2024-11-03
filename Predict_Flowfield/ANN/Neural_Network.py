import torch.nn as nn
import torch.nn.functional as F


class ANN_flowfield(nn.Module):
    """
    Artificial Neural Network (ANN) for generating a 128x128 flow field image from a 12-dimensional input,
    with configurable hidden layers, dropout, and a fixed Leaky ReLU activation function.

    Parameters:
        hidden_sizes (int): Size of each hidden layer.
        hidden_number (int): Number of hidden layers.
        dropout_rate (float): Dropout rate applied to each hidden layer.
        input_size (int): Size of the input layer, default is 12.
        output_size (int): Size of the output layer, default is 128*128.
    """

    OUTPUT_DIM = 128  # Default output image dimension

    def __init__(self, hidden_sizes, hidden_number, dropout_rate, input_size=12, output_size=OUTPUT_DIM * OUTPUT_DIM):
        super(ANN_flowfield, self).__init__()
        self.layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.hidden_sizes = [hidden_sizes] * hidden_number

        # Construct hidden layers with BatchNorm and Dropout
        previous_size = input_size
        for hidden_size in self.hidden_sizes:
            self.layers.append(nn.Linear(previous_size, hidden_size))   # Linear layer
            self.bn_layers.append(nn.BatchNorm1d(hidden_size))          # Batch normalization layer
            self.dropouts.append(nn.Dropout(dropout_rate))              # Dropout layer
            previous_size = hidden_size

        # Output layer
        self.output_layer = nn.Linear(previous_size, output_size)

    def forward(self, x):
        """
        Forward pass of the ANN model.

        Parameters:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor reshaped to (batch_size, 1, 128, 128).
        """
        # Pass through each hidden layer with Leaky ReLU activation
        for layer, bn_layer, dropout in zip(self.layers, self.bn_layers, self.dropouts):
            x = F.leaky_relu(bn_layer(layer(x)), negative_slope=0.2)
            x = dropout(x)

        # Output layer and reshape to (batch_size, 1, 128, 128)
        x = self.output_layer(x)
        x = x.view(-1, 1, self.OUTPUT_DIM, self.OUTPUT_DIM)

        return x
