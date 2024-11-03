import torch
import torch.nn as nn
import numpy as np

def train(epochs, save_interval, model_ANN, train_dataloader, test_dataloader, device, optimizer_ANN,
          criterion_ANN, scheduler_ANN, iMach_mean, iMach_std, train_dataloader_for_eval,
          test_dataloader_for_eval, FOLD_INDEX, BATCH_SIZE, ANN_DROPOUT_RATE, HIDDEN_SIZES, HIDDEN_NUMBER):
    """
    Train the ANN model for a specified number of epochs, logging training and test losses,
    as well as Mean Absolute Error (MAE) for both datasets. Saves model checkpoints and loss metrics.

    Parameters:
        epochs (int): Number of training epochs.
        save_interval (int): Interval for saving model checkpoints.
        model_ANN (torch.nn.Module): ANN model to be trained.
        train_dataloader (DataLoader): Dataloader for the training set.
        test_dataloader (DataLoader): Dataloader for the test set.
        device (torch.device): Device for computation (CPU or GPU).
        optimizer_ANN (torch.optim.Optimizer): Optimizer for the ANN model.
        criterion_ANN (torch.nn.Module): Loss function for the ANN model.
        scheduler_ANN (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        iMach_mean (torch.Tensor): Mean for normalization.
        iMach_std (torch.Tensor): Standard deviation for normalization.
        train_dataloader_for_eval (DataLoader): Dataloader for evaluating the training set.
        test_dataloader_for_eval (DataLoader): Dataloader for evaluating the test set.
        FOLD_INDEX (int): Index for cross-validation fold.
        BATCH_SIZE (int): Batch size for training.
        ANN_DROPOUT_RATE (float): Dropout rate for ANN.
        HIDDEN_SIZES (int): Number of units in each hidden layer.
        HIDDEN_NUMBER (int): Number of hidden layers in the ANN.
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
        for inputs, images, _ in train_dataloader:
            inputs, images = inputs.to(device), images.to(device)
            images = images.view(-1, 1, 128, 128)

            optimizer_ANN.zero_grad()
            fake_images = model_ANN(inputs.float())
            g_loss = criterion_ANN(fake_images, images)
            g_loss.backward()
            optimizer_ANN.step()

            scheduler_ANN.step()

            # Accumulate training loss
            total_loss_train_ANN += g_loss.item() * inputs.size(0)
            total_train_samples += inputs.size(0)

        # Evaluation step
        model_ANN.eval()
        total_loss_test_ANN = 0
        total_test_samples = 0

        with torch.no_grad():
            for inputs, images, _ in test_dataloader_for_eval:
                inputs, images = inputs.to(device), images.to(device)
                outputs = model_ANN(inputs.float())
                test_loss = criterion_ANN(outputs, images.view(-1, 1, 128, 128))
                total_loss_test_ANN += test_loss.item() * inputs.size(0)
                total_test_samples += inputs.size(0)

                # Denormalize and calculate MAE for the test set
                images = images * iMach_std + iMach_mean
                outputs = outputs * iMach_std + iMach_mean
                test_mae = mae_loss(outputs, images.view(-1, 1, 128, 128))

            # Calculate MAE for the training set
            for inputs, images, _ in train_dataloader_for_eval:
                inputs, images = inputs.to(device), images.to(device)
                outputs = model_ANN(inputs.float())
                images = images * iMach_std + iMach_mean
                outputs = outputs * iMach_std + iMach_mean
                train_mae = mae_loss(outputs, images.view(-1, 1, 128, 128))

        # Print epoch summary
        print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {total_loss_train_ANN / total_train_samples:.4f}, '
              f'Test Loss: {total_loss_test_ANN / total_test_samples:.4f}, '
              f'Train MAE: {train_mae:.4f}, Test MAE: {test_mae:.4f}')

        # Append metrics for tracking
        loss_train_ANN_all.append(total_loss_train_ANN / total_train_samples)
        loss_test_ANN_all.append(total_loss_test_ANN / total_test_samples)
        mae_train_all.append(train_mae.item())
        mae_test_all.append(test_mae.item())

        # Save intermediate results
        if (epoch + 1) % save_interval == 0:
            from Plot_dataloader_Figure import plot_pre_dataloader_figure
            plot_pre_dataloader_figure(
                test_dataloader, iMach_mean, iMach_std,
                f'Train_{BATCH_SIZE}_{ANN_DROPOUT_RATE}_{HIDDEN_SIZES}_{HIDDEN_NUMBER}/{FOLD_INDEX}/epoch_{epoch + 1}_test_fig.png',
                model_ANN
            )

    # Save model and loss metrics
    torch.save(model_ANN, f'Train_{BATCH_SIZE}_{ANN_DROPOUT_RATE}_{HIDDEN_SIZES}_{HIDDEN_NUMBER}/{FOLD_INDEX}/model_ANN.pt')
    np.savetxt(f"Train_{BATCH_SIZE}_{ANN_DROPOUT_RATE}_{HIDDEN_SIZES}_{HIDDEN_NUMBER}/{FOLD_INDEX}/loss_train_ANN", np.array(loss_train_ANN_all))
    np.savetxt(f"Train_{BATCH_SIZE}_{ANN_DROPOUT_RATE}_{HIDDEN_SIZES}_{HIDDEN_NUMBER}/{FOLD_INDEX}/loss_test_ANN", np.array(loss_test_ANN_all))
    np.savetxt(f"Train_{BATCH_SIZE}_{ANN_DROPOUT_RATE}_{HIDDEN_SIZES}_{HIDDEN_NUMBER}/{FOLD_INDEX}/mae_train", np.array(mae_train_all))
    np.savetxt(f"Train_{BATCH_SIZE}_{ANN_DROPOUT_RATE}_{HIDDEN_SIZES}_{HIDDEN_NUMBER}/{FOLD_INDEX}/mae_test", np.array(mae_test_all))
