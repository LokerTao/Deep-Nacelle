import torch.nn as nn
import torch.nn.functional as F


class DNN_flowfield(nn.Module):
    """
    Deep Neural Network (DNN) for generating a 128x128 flow field image from a 12-dimensional input.

    Parameters:
        dropout_rate (float): Dropout rate to be applied to upsampling layers.

    Architecture:
        - Three fully connected (FC) layers to expand the input features.
        - Series of transposed convolutional (deconvolution) layers for upsampling to 128x128 resolution.
    """

    def __init__(self, dropout_rate):
        super(DNN_flowfield, self).__init__()

        # Fully connected layers for feature expansion
        self.fc1 = nn.Linear(12, 128)  # Expands input to 128 features
        self.fc2 = nn.Linear(128, 512)  # Expands further to 512 features
        self.fc3 = nn.Linear(512, 512)  # Maintains 512 features for spatial processing

        # Batch normalization for fully connected layers
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(512)

        # Upsampling layers (transposed convolutional layers with BatchNorm and dropout)
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate)
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate)
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate)
        )
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate)
        )
        self.deconv5 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate)
        )
        self.deconv6 = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate)
        )
        self.deconv7 = nn.Sequential(
            nn.ConvTranspose2d(8, 1, kernel_size=3, stride=2, padding=1, output_padding=1)  # Final output layer
        )

    def forward(self, x):
        """
        Forward pass of the DNN model.

        Parameters:
            x (torch.Tensor): Input tensor of shape (batch_size, 12).

        Returns:
            torch.Tensor: Output tensor reshaped to (batch_size, 1, 128, 128).
        """

        # Fully connected layers with Leaky ReLU activation and Batch Normalization
        x = F.leaky_relu(self.bn1(self.fc1(x)), 0.2)  # Expands to 128 features
        x = F.leaky_relu(self.bn2(self.fc2(x)), 0.2)  # Expands to 512 features
        x = F.leaky_relu(self.bn3(self.fc3(x)), 0.2)  # Maintains 512 features

        # Reshape for convolutional layers
        x = x.view(-1, 512, 1, 1)  # Reshapes to (batch, 512, 1, 1) for upsampling

        # Transposed convolution layers for upsampling
        x = self.deconv1(x)  # (batch, 256, 2, 2)
        x = self.deconv2(x)  # (batch, 128, 4, 4)
        x = self.deconv3(x)  # (batch, 64, 8, 8)
        x = self.deconv4(x)  # (batch, 32, 16, 16)
        x = self.deconv5(x)  # (batch, 16, 32, 32)
        x = self.deconv6(x)  # (batch, 8, 64, 64)
        x = self.deconv7(x)  # (batch, 1, 128, 128) - Final output

        return x
