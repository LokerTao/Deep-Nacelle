import torch.nn as nn
import torch.nn.functional as F


class ANN_flowfield(nn.Module):
    """
    Artificial Neural Network (ANN) for predicting flow field values with configurable hidden layers,
    dropout, and a fixed Leaky ReLU activation function.

    Parameters:
        hidden_sizes (int): Size of each hidden layer.
        hidden_number (int): Number of hidden layers.
        dropout_rate (float): Dropout rate applied to each hidden layer.
        input_size (int): Size of the input layer, default is 12.
        output_size (int): Size of the output layer, default is 1 (single prediction value per sample).
    """

    def __init__(self, hidden_sizes, hidden_number, dropout_rate, input_size=12, output_size=1):
        super(ANN_flowfield, self).__init__()
        self.layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.hidden_sizes = [hidden_sizes] * hidden_number

        # Construct hidden layers with BatchNorm and Dropout
        previous_size = input_size
        for hidden_size in self.hidden_sizes:
            self.layers.append(nn.Linear(previous_size, hidden_size))
            self.bn_layers.append(nn.BatchNorm1d(hidden_size))
            self.dropouts.append(nn.Dropout(dropout_rate))
            previous_size = hidden_size

        # Output layer
        self.output_layer = nn.Linear(previous_size, output_size)

    def forward(self, x):
        """
        Forward pass of the ANN model.

        Parameters:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
        """
        # Pass through each hidden layer with Leaky ReLU activation
        for layer, bn_layer, dropout in zip(self.layers, self.bn_layers, self.dropouts):
            x = F.leaky_relu(bn_layer(layer(x)), negative_slope=0.2)
            x = dropout(x)

        # Output layer
        x = self.output_layer(x)

        return x  # Shape (batch_size, 1)
