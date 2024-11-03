import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import argparse
from pathlib import Path

# ===================== Configurations =====================

# Training parameters
BATCH_SIZE = 1080  # Possible batch sizes: 16, 32, 64, 1080
DNN_DROPOUT_RATE = 0  # Possible dropout rates for DNN: 0, 0.1, 0.2, 0.3
FOLD_INDEX = 1  # Indicates the K-fold index, where K = 10; FOLD_INDEX ranges from 1 to 10

# Model path for Predict Flowfield model
model_g_path = f'../../Predict_Flowfield/DNN/Train_1080_0/{FOLD_INDEX}/model_DNN.pt'  # Load model with consistent FOLD_INDEX for dataset compatibility

# Command-line argument parsing
parser = argparse.ArgumentParser(description="Training script for DNN model")
parser.add_argument("--n_epochs", type=int, default=1000, help="Number of training epochs")
parser.add_argument("--train_batch_size", type=int, default=BATCH_SIZE, help="Training batch size")
parser.add_argument("--test_batch_size", type=int, default=128, help="Test batch size")
parser.add_argument("--lr_DNN", type=float, default=0.002, help="Learning rate for DNN")
parser.add_argument("--b1", type=float, default=0.9, help="Adam optimizer: first-order momentum decay")
parser.add_argument("--b2", type=float, default=0.999, help="Adam optimizer: second-order momentum decay")
parser.add_argument("--scheduler_step_size_DNN", type=int, default=500, help="Scheduler step size for DNN")
parser.add_argument("--scheduler_gamma_DNN", type=float, default=0.1, help="Scheduler gamma value")
parser.add_argument("--DNN_dropout_rate", type=float, default=DNN_DROPOUT_RATE, help="Dropout rate for DNN")
opt = parser.parse_args()

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===================== Directory Setup =====================

# Ensure output directory exists
output_dir = Path(f"Train_{BATCH_SIZE}_{DNN_DROPOUT_RATE}/{FOLD_INDEX}")
output_dir.mkdir(parents=True, exist_ok=True)

# ===================== Data Loading =====================

train_dataset = torch.load(f"../../DataProcess/data_fold/{FOLD_INDEX}/train_dataset.pth")
test_dataset = torch.load(f"../../DataProcess/data_fold/{FOLD_INDEX}/test_dataset.pth")

train_dataloader = DataLoader(train_dataset, batch_size=opt.train_batch_size, shuffle=True, drop_last=True)

# DataLoader for evaluation (loads the entire dataset in one batch)
train_dataloader_for_eval = DataLoader(train_dataset, batch_size=len(train_dataset))
test_dataloader_for_eval = DataLoader(test_dataset, batch_size=len(test_dataset))

# ===================== Model Setup =====================

from Neural_Network import DNN_flowfield, DNN_cd

# Initialize the DNN model
model_DNN = DNN_cd(label_dim=12, dropout_rate=opt.DNN_dropout_rate).to(device)

# Define loss function and optimizer
criterion_DNN = nn.MSELoss()  # Mean Squared Error loss
optimizer_DNN = optim.Adam(model_DNN.parameters(), lr=opt.lr_DNN, betas=(opt.b1, opt.b2))
scheduler_DNN = StepLR(optimizer_DNN, step_size=opt.scheduler_step_size_DNN, gamma=opt.scheduler_gamma_DNN)


# ===================== Initialize Weights =====================

def weights_init_normal(m):
    """
    Applies initial weights to the model layers:

    - Conv layers: Normal distribution with mean=0.0, std=0.02.
    - BatchNorm2d layers: Normal distribution with mean=1.0, std=0.02 for weights, and biases set to zero.
    """
    classname = m.__class__.__name__
    if "Conv" in classname:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm2d" in classname:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


# Apply weight initialization to the model
model_DNN.apply(weights_init_normal)

# ===================== Training =====================

from Train_Process import train

# Start training
train(opt.n_epochs, model_DNN, train_dataloader, device,
      optimizer_DNN, criterion_DNN, scheduler_DNN, train_dataloader_for_eval,
      test_dataloader_for_eval, FOLD_INDEX, BATCH_SIZE, DNN_DROPOUT_RATE, model_g_path)
