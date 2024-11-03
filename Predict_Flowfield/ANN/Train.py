import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import argparse
from pathlib import Path

# ===================== Configurations =====================

# Training parameters
BATCH_SIZE = 1080  # Possible values: 16, 32, 64, 1080
ANN_DROPOUT_RATE = 0.1  # Possible values: 0, 0.1, 0.2, 0.3
HIDDEN_SIZES = 1024  # Possible values: 128, 256, 512, 1024
HIDDEN_NUMBER = 4  # Possible values: 2, 3, 4, 5
FOLD_INDEX = 1  # Order of K-fold cross-validation, with K=10 and index range 1 to 10

# Command-line argument parsing
parser = argparse.ArgumentParser(description="Training script for neural network model")
parser.add_argument("--n_epochs", type=int, default=1000, help="Number of training epochs")
parser.add_argument("--save_interval", type=int, default=1000, help="Save interval for model checkpoints")
parser.add_argument("--train_batch_size", type=int, default=BATCH_SIZE, help="Training batch size")
parser.add_argument("--test_batch_size", type=int, default=128, help="Test batch size")
parser.add_argument("--lr_ANN", type=float, default=0.002, help="Learning rate for ANN")
parser.add_argument("--b1", type=float, default=0.9, help="Adam optimizer: first-order momentum decay")
parser.add_argument("--b2", type=float, default=0.999, help="Adam optimizer: second-order momentum decay")
parser.add_argument("--scheduler_step_size_ANN", type=int, default=500, help="Scheduler step size")
parser.add_argument("--scheduler_gamma_ANN", type=float, default=0.1, help="Scheduler gamma value")
parser.add_argument("--ANN_dropout_rate", type=float, default=ANN_DROPOUT_RATE, help="Dropout rate for ANN")
parser.add_argument("--hidden_sizes", type=int, default=HIDDEN_SIZES, help="Hidden layer size for ANN")
parser.add_argument("--hidden_number", type=int, default=HIDDEN_NUMBER, help="Number of hidden layers in ANN")
opt = parser.parse_args()

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===================== Directory Setup =====================

# Ensure output directory exists
output_dir = Path(f"Train_{BATCH_SIZE}_{ANN_DROPOUT_RATE}_{HIDDEN_SIZES}_{HIDDEN_NUMBER}/{FOLD_INDEX}")
output_dir.mkdir(parents=True, exist_ok=True)

# ===================== Data Loading =====================

train_dataset = torch.load(f"../../DataProcess/data_fold/{FOLD_INDEX}/train_dataset.pth")
test_dataset = torch.load(f"../../DataProcess/data_fold/{FOLD_INDEX}/test_dataset.pth")

train_dataloader = DataLoader(train_dataset, batch_size=opt.train_batch_size, shuffle=True, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=opt.test_batch_size)

# DataLoader for evaluation (loads the entire dataset in one batch)
train_dataloader_for_eval = DataLoader(train_dataset, batch_size=len(train_dataset))
test_dataloader_for_eval = DataLoader(test_dataset, batch_size=len(test_dataset))

# Mean and standard deviation for normalization
iMach_mean = torch.from_numpy(np.load(f"../../DataProcess/data_fold/{FOLD_INDEX}/iMach_mean.npy")).to(device)
iMach_std = torch.from_numpy(np.load(f"../../DataProcess/data_fold/{FOLD_INDEX}/iMach_std.npy")).to(device)

# ===================== Visualization =====================

from Plot_dataloader_Figure import plot_dataloader_figure

# Plot and save the figure for data visualization
plot_dataloader_figure(test_dataloader, iMach_mean, iMach_std, output_dir / 'test_fig.png')

# ===================== Model Setup =====================

from Neural_Network import ANN_flowfield

# Initialize the model with given hyperparameters
model_ANN = ANN_flowfield(opt.hidden_sizes, opt.hidden_number, opt.ANN_dropout_rate).to(device)

# Define loss function and optimizer
criterion_ANN = nn.MSELoss()  # Mean Squared Error loss
optimizer_ANN = optim.Adam(model_ANN.parameters(), lr=opt.lr_ANN, betas=(opt.b1, opt.b2))
scheduler_ANN = StepLR(optimizer_ANN, step_size=opt.scheduler_step_size_ANN, gamma=opt.scheduler_gamma_ANN)


# ===================== Initialize Weights =====================

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

# Apply weight initialization
model_ANN.apply(weights_init_normal)

# ===================== Training =====================

from Train_Process import train

# Run the training process with specified parameters and dataloaders
train(opt.n_epochs, opt.save_interval, model_ANN, train_dataloader, test_dataloader, device,
      optimizer_ANN, criterion_ANN, scheduler_ANN, iMach_mean, iMach_std, train_dataloader_for_eval,
      test_dataloader_for_eval, FOLD_INDEX,BATCH_SIZE,ANN_DROPOUT_RATE,HIDDEN_SIZES,HIDDEN_NUMBER)
