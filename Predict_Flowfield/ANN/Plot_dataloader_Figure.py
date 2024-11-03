import math
import matplotlib.pyplot as plt
import numpy as np
import os
import torch

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_dataloader_figure(dataloader,iMach_mean,iMach_std,output_filepath):

    for inputs, images,cd in dataloader:

        images = images.to(device)

        images = images*iMach_std+iMach_mean

        ncols = 8

        nrows = math.ceil(len(dataloader.dataset)/ncols)

        figsize_width_per_subplot = 4

        figsize_height_per_subplot = 3

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(figsize_width_per_subplot*ncols, figsize_height_per_subplot*nrows), gridspec_kw={'wspace': 0.025, 'hspace': 0.025})

        axes_flat = axes.flat

        for i, data in enumerate(images):
            data = data.cpu()

            data = data.squeeze()

            contourf = axes_flat[i].contourf(data, levels=np.linspace(0, 2, 21), extend='both')

            axes_flat[i].axis('off')

        for j in range(i + 1, nrows*ncols):
            axes_flat[j].axis('off')

        fig.colorbar(contourf, ax=axes.ravel().tolist(), extend='both', fraction=0.01)

        plt.savefig(output_filepath)
        plt.close()

def plot_pre_dataloader_figure(dataloader,iMach_mean,iMach_std,output_filepath,model_DNN):

    for inputs, images,cd in dataloader:

        pre_images = model_DNN(inputs.to(device).float())

        pre_images = pre_images * iMach_std + iMach_mean

        ncols = 8

        nrows = math.ceil(len(dataloader.dataset)/ncols)

        figsize_width_per_subplot = 4

        figsize_height_per_subplot = 3

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(figsize_width_per_subplot*ncols, figsize_height_per_subplot*nrows), gridspec_kw={'wspace': 0.025, 'hspace': 0.025})

        axes_flat = axes.flat

        for i, data in enumerate(pre_images):
            data = data.detach().cpu()

            data = data.squeeze()

            contourf = axes_flat[i].contourf(data, levels=np.linspace(0, 2, 21), extend='both')

            axes_flat[i].axis('off')

        for j in range(i + 1, nrows*ncols):
            axes_flat[j].axis('off')

        fig.colorbar(contourf, ax=axes.ravel().tolist(), extend='both', fraction=0.01)

        plt.savefig(output_filepath)
        plt.close()