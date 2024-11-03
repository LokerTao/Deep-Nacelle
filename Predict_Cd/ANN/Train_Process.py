import torch
import torch.nn as nn
import numpy as np

def train(epochs, model_ANN, train_dataloader, device, optimizer_ANN,
          criterion_ANN, scheduler_ANN, train_dataloader_for_eval,
          test_dataloader_for_eval, FOLD_INDEX, BATCH_SIZE, ANN_DROPOUT_RATE, HIDDEN_SIZES, HIDDEN_NUMBER):
    """
    Train the ANN model for a specified number of epochs, logging training and test losses,
    as well as Mean Absolute Error (MAE) for both datasets. Saves model checkpoints and loss metrics.

    Parameters:
        epochs (int): Number of training epochs.
        model_ANN (torch.nn.Module): ANN model to be trained.
        train_dataloader (DataLoader): Dataloader for the training set.
        device (torch.device): Device for computation (CPU or GPU).
        optimizer_ANN (torch.optim.Optimizer): Optimizer for the ANN model.
        criterion_ANN (torch.nn.Module): Loss function for the ANN model.
        scheduler_ANN (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        train_dataloader_for_eval (DataLoader): Dataloader for evaluating the training set.
        test_dataloader_for_eval (DataLoader): Dataloader for evaluating the test set.
        FOLD_INDEX (int): Identifier for the current fold in K-fold cross-validation.
        BATCH_SIZE (int): Batch size used during training.
        ANN_DROPOUT_RATE (float): Dropout rate applied to the ANN model.
        HIDDEN_SIZES (int): Size of hidden layers in the ANN model.
        HIDDEN_NUMBER (int): Number of hidden layers in the ANN model.

    Returns:
        None
    """

    # Initialize loss functions and storage for metrics
    mae_loss = nn.L1Loss()
    loss_train_ANN_all = []
    loss_test_ANN_all = []
    mae_train_all = []
    mae_test_all = []

    # Training loop
    for epoch in range(epochs):
        model_ANN.train()  # Set model to training mode
        total_loss_train_ANN = 0
        total_train_samples = 0

        # Training step
        for inputs, images, cd in train_dataloader:
            inputs, images, cd = inputs.to(device), images.to(device), cd.to(device)

            optimizer_ANN.zero_grad()
            cd_pre = model_ANN(inputs.float())
            g_loss = criterion_ANN(cd_pre, cd)
            g_loss.backward()
            optimizer_ANN.step()
            scheduler_ANN.step()  # Update learning rate

            # Accumulate training loss
            total_loss_train_ANN += g_loss.item() * inputs.size(0)
            total_train_samples += inputs.size(0)

        # Evaluation step
        model_ANN.eval()
        total_loss_test_ANN = 0
        total_test_samples = 0

        with torch.no_grad():
            # Evaluate on test set
            for inputs, images, cd in test_dataloader_for_eval:
                inputs, images, cd = inputs.to(device), images.to(device), cd.to(device)
                outputs = model_ANN(inputs.float())
                test_loss = criterion_ANN(outputs, cd)
                total_loss_test_ANN += test_loss.item() * inputs.size(0)
                total_test_samples += inputs.size(0)

                # Scale outputs and ground truth for MAE calculation
                outputs = outputs * 0.205
                cd = cd * 0.205
                test_mae = mae_loss(outputs, cd)

            # Calculate MAE for train set
            for inputs, images, cd in train_dataloader_for_eval:
                inputs, images, cd = inputs.to(device), images.to(device), cd.to(device)
                outputs = model_ANN(inputs.float())
                outputs = outputs * 0.205
                cd = cd * 0.205
                train_mae = mae_loss(outputs, cd)

        # Print epoch summary
        print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {total_loss_train_ANN / total_train_samples:.4f}, '
              f'Test Loss: {total_loss_test_ANN / total_test_samples:.4f}, '
              f'Train MAE: {train_mae:.4f}, Test MAE: {test_mae:.4f}')

        # Append metrics for tracking
        loss_train_ANN_all.append(total_loss_train_ANN / total_train_samples)
        loss_test_ANN_all.append(total_loss_test_ANN / total_test_samples)
        mae_train_all.append(train_mae.item())
        mae_test_all.append(test_mae.item())

    # Save model and loss metrics
    torch.save(model_ANN, f'Train_{BATCH_SIZE}_{ANN_DROPOUT_RATE}_{HIDDEN_SIZES}_{HIDDEN_NUMBER}/{FOLD_INDEX}/model_ANN.pt')
    np.savetxt(f"Train_{BATCH_SIZE}_{ANN_DROPOUT_RATE}_{HIDDEN_SIZES}_{HIDDEN_NUMBER}/{FOLD_INDEX}/loss_train_ANN", np.array(loss_train_ANN_all))
    np.savetxt(f"Train_{BATCH_SIZE}_{ANN_DROPOUT_RATE}_{HIDDEN_SIZES}_{HIDDEN_NUMBER}/{FOLD_INDEX}/loss_test_ANN", np.array(loss_test_ANN_all))
    np.savetxt(f"Train_{BATCH_SIZE}_{ANN_DROPOUT_RATE}_{HIDDEN_SIZES}_{HIDDEN_NUMBER}/{FOLD_INDEX}/mae_train", np.array(mae_train_all))
    np.savetxt(f"Train_{BATCH_SIZE}_{ANN_DROPOUT_RATE}_{HIDDEN_SIZES}_{HIDDEN_NUMBER}/{FOLD_INDEX}/mae_test", np.array(mae_test_all))
