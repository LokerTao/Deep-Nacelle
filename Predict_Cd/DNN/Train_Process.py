import torch
import torch.nn as nn
import numpy as np

def train(epochs, model_DNN, train_dataloader, device, optimizer_DNN,
          criterion_DNN, scheduler_DNN, train_dataloader_for_eval,
          test_dataloader_for_eval, FOLD_INDEX, BATCH_SIZE, DNN_DROPOUT_RATE, model_g_path):
    """
    Train the DNN model for a specified number of epochs, logging training and test losses,
    as well as Mean Absolute Error (MAE) for both datasets. Saves model checkpoints and loss metrics.

    Parameters:
        epochs (int): Number of training epochs.
        model_DNN (torch.nn.Module): Neural network model for predicting cd values.
        train_dataloader (DataLoader): Dataloader for the training set.
        test_dataloader_for_eval (DataLoader): Dataloader for the test set (for evaluation).
        device (torch.device): Device for computation (CPU or GPU).
        optimizer_DNN (torch.optim.Optimizer): Optimizer for the DNN model.
        criterion_DNN (torch.nn.Module): Loss function for the DNN model.
        scheduler_DNN (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        train_dataloader_for_eval (DataLoader): Dataloader for evaluating the training set.
        FOLD_INDEX (int): Fold index for K-fold cross-validation.
        BATCH_SIZE (int): Batch size used for training.
        DNN_DROPOUT_RATE (float): Dropout rate used in the DNN model.
        model_g_path (str): Path to the pre-trained model used for generating inputs.

    Returns:
        None
    """

    # Initialize loss functions and storage for metrics
    mae_loss = nn.L1Loss()
    loss_train_DNN_all = []
    loss_test_DNN_all = []
    mae_train_all = []
    mae_test_all = []

    # Load pre-trained model_g
    model_g = torch.load(model_g_path)
    model_g.eval()  # Set model_g to evaluation mode

    # Training loop
    for epoch in range(epochs):
        model_DNN.train()  # Set model_DNN to training mode
        total_loss_train_DNN = 0
        total_train_samples = 0

        # Training step
        for inputs, images, cd in train_dataloader:
            inputs, images, cd = inputs.to(device), images.to(device), cd.to(device)

            optimizer_DNN.zero_grad()
            fake_images = model_g(inputs.float())
            cd_pre = model_DNN(fake_images, inputs)
            DNN_loss = criterion_DNN(cd_pre, cd)
            DNN_loss.backward()
            optimizer_DNN.step()
            scheduler_DNN.step()  # Step the learning rate scheduler

            # Accumulate training loss
            total_loss_train_DNN += DNN_loss.item() * inputs.size(0)
            total_train_samples += inputs.size(0)

        # Evaluation step
        model_DNN.eval()  # Set model_DNN to evaluation mode
        total_loss_test_DNN = 0
        total_test_samples = 0

        with torch.no_grad():
            for inputs, images, cd in test_dataloader_for_eval:
                inputs, images, cd = inputs.to(device), images.to(device), cd.to(device)
                outputs = model_DNN(model_g(inputs.float()), inputs)
                test_loss = criterion_DNN(outputs, cd)
                total_loss_test_DNN += test_loss.item() * inputs.size(0)
                total_test_samples += inputs.size(0)

                # Denormalize and calculate MAE for test set
                outputs = outputs * 0.205  # Scale output for MAE calculation
                cd = cd * 0.205  # Scale ground truth for MAE calculation
                test_mae = mae_loss(outputs, cd)

            # Calculate MAE for train set
            for inputs, images, cd in train_dataloader_for_eval:
                inputs, images, cd = inputs.to(device), images.to(device), cd.to(device)
                outputs = model_DNN(model_g(inputs.float()), inputs)
                outputs = outputs * 0.205
                cd = cd * 0.205
                train_mae = mae_loss(outputs, cd)

        # Print epoch summary
        print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {total_loss_train_DNN / total_train_samples:.4f}, '
              f'Test Loss: {total_loss_test_DNN / total_test_samples:.4f}, '
              f'Train MAE: {train_mae:.4f}, Test MAE: {test_mae:.4f}')

        # Append metrics for tracking
        loss_train_DNN_all.append(total_loss_train_DNN / total_train_samples)
        loss_test_DNN_all.append(total_loss_test_DNN / total_test_samples)
        mae_train_all.append(train_mae.item())
        mae_test_all.append(test_mae.item())

    # Save model and loss metrics
    torch.save(model_DNN, f'Train_{BATCH_SIZE}_{DNN_DROPOUT_RATE}/{FOLD_INDEX}/model_DNN.pt')
    np.savetxt(f"Train_{BATCH_SIZE}_{DNN_DROPOUT_RATE}/{FOLD_INDEX}/loss_train_DNN", np.array(loss_train_DNN_all))
    np.savetxt(f"Train_{BATCH_SIZE}_{DNN_DROPOUT_RATE}/{FOLD_INDEX}/loss_test_DNN", np.array(loss_test_DNN_all))
    np.savetxt(f"Train_{BATCH_SIZE}_{DNN_DROPOUT_RATE}/{FOLD_INDEX}/mae_train", np.array(mae_train_all))
    np.savetxt(f"Train_{BATCH_SIZE}_{DNN_DROPOUT_RATE}/{FOLD_INDEX}/mae_test", np.array(mae_test_all))
