import torch.nn as nn
import torch.nn.functional as F
import torch

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


class DNN_cd(nn.Module):
    def __init__(self, label_dim, dropout_rate=0.1):
        super(DNN_cd, self).__init__()

        self.label_dim = label_dim
        # Fully connected layer to expand label_dim to match image size (128x128)
        self.label_fc = nn.Linear(label_dim, 128 * 128)  # Output dimension: [batch_size, 128 * 128]

        # Convolutional layers for processing concatenated input and label
        self.model = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=4, stride=2, padding=1),  # Output size: 64x64
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout_rate),

            nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1),  # Output size: 32x32
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout_rate),

            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),  # Output size: 16x16
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout_rate),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # Output size: 8x8
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout_rate),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # Output size: 4x4
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout_rate),

            nn.Conv2d(128, 1, kernel_size=4, stride=1),  # Output size: 1x1
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        """
        Forward pass of the DNN_cd model.

        Parameters:
            x (torch.Tensor): Input tensor with shape [batch_size, img_channel, 128, 128].
            labels (torch.Tensor): Label tensor with shape [batch_size, label_dim].

        Returns:
            torch.Tensor: Output tensor with shape [batch_size, 1].
        """
        # Expand labels to match input spatial dimensions by fully connecting to 128x128
        labels = self.label_fc(labels).view(x.size(0), 1, 128, 128)

        # Concatenate the input and expanded label along the channel dimension
        x = torch.cat((x, labels), dim=1)  # Concatenated shape: [batch_size, 2, 128, 128]

        # Pass concatenated tensor through the model
        output = self.model(x)
        return output.view(-1, 1)  # Flatten to [batch_size, 1]

