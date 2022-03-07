#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Script to train a SRGAN"""

__author__ = "Maximiliano Bove"
__email__ = "maxibove13@gmail.com"
__status__ = "Development"
__date__ = "02/22"


# Built-in modules
import os

# Third party modules
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from sklearn.model_selection import KFold
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
import yaml
# from tqdm import tqdm

# Local modules
from data.data_utils import ImageDataset
from models.model import Generator, Discriminator, VGGLoss, train_discriminator, train_generator
from utils import load_checkpoint, save_checkpoint

# read yaml file
with open('config.yaml') as file:
    config = yaml.safe_load(file)
# By default hyperparameters are defined by config.yaml
learning_rate = config['train']['learning_rate']
num_epochs = config['train']['num_epochs']
batch_size = config['train']['batch_size']
num_workers = config['train']['num_workers']

def train_srgan(learning_rate, num_epochs, batch_size, num_workers):

    # Define device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {torch.cuda.get_device_name()}")

    # Import High Resolution images (dataset)
    print("Importing dataset...")
    dataset = ImageDataset(root_dir=os.path.join(config['data']['rootdir'], config['data']['dataset'], 'train'))
    print(f"{len(dataset)} training samples from {dataset.root_dir}")



    # Initialize models and send them to device
    print('Initializing Generator and Discriminator...')
    gen = Generator(in_channels=config['data']['img_channels']).to(device)
    disc = Discriminator(in_channels=config['data']['img_channels']).to(device)

    print('Defining losses and optimizers...')
    # Define losses
    bce = nn.BCEWithLogitsLoss()
    vgg_loss_fun = VGGLoss()
    mse = nn.MSELoss()

    # Define optimizers
    opt_gen = optim.Adam(gen.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=learning_rate, betas=(0.9, 0.999))

    # Load checkpoint
    if config['models']['load']:
        print(f"Loading pretrained model from {config['models']['rootdir']}")
        load_checkpoint(
            os.path.join(config['models']['rootdir'], config['models']['gen']),
            gen,
            opt_gen,
            learning_rate,
            device
        )
        load_checkpoint(
            os.path.join(config['models']['rootdir'], config['models']['disc']), disc, opt_disc, learning_rate, 
            device
        )

    kfold = config['validation']['kfold']
    if kfold:
        n_splits = config['validation']['n_splits']
        kfold = KFold(n_splits=n_splits, shuffle=True)
        psnr = np.zeros((n_splits, len(dataset)//n_splits//batch_size + 1, batch_size))
        ssim = np.zeros((n_splits, len(dataset)//n_splits//batch_size + 1, batch_size))

    # Iterate over kfolds only if kfold is on.
    for fold, (train_idx, test_idx) in enumerate(kfold.split(dataset)) if kfold else enumerate([(0,0)]):

        if kfold:
            train_subsampler = SubsetRandomSampler(train_idx)
            test_subsampler = SubsetRandomSampler(test_idx)
            # Load data
            print(f"\nSpliting and loading datasets [{fold+1}/{n_splits}]...")
            trainloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True, num_workers=num_workers, sampler=train_subsampler)
            testloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True, num_workers=num_workers, sampler=test_subsampler)
        else:
            print(f"\nLoading dataset...")
            trainloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True, num_workers=num_workers)


        # Training loop
        print(f"Starting training: \n")
        if kfold:
            print(
                f"  k-fold: {fold+1}/{n_splits}\n"
                f"  Training samples on this fold: {len(train_subsampler)}\n"
                f"  Testing samples on this fold: {len(test_subsampler)}\n"
                f"  Number of epochs: {num_epochs}\n"
                f"  Mini batch size: {batch_size}\n"
                f"  Number of batches: {len(trainloader)}\n"
                f"  Learning rate: {learning_rate})\n"
            )
        else:
            print(
                f"  Training samples: {len(dataset)}\n"
                f"  Number of epochs: {num_epochs}\n"
                f"  Mini batch size: {batch_size}\n"
                f"  Number of batches: {len(trainloader)}\n"
                f"  Learning rate: {learning_rate})\n"
            )
        loss_disc = []
        loss_gen = []

        fig, ax = plt.subplots(figsize=(10,6), dpi= 80)

        for epoch in range(num_epochs):
            for idx, (low_res, high_res) in enumerate(trainloader):

                # Send images to device
                high_res = high_res.to(device)
                low_res = low_res.to(device)

                # Generate fake (high_res) image from low_res
                fake = gen(low_res)

                loss_disc_e = train_discriminator(disc, opt_disc, fake, high_res, bce)
                loss_gen_e = train_generator(disc, opt_gen, fake, high_res, vgg_loss_fun, mse, bce)

                # At the end of every epoch
                if idx == batch_size-1:

                    # Append current epoch loss to list of losses
                    loss_disc.append(float(loss_disc_e.detach().cpu()))
                    loss_gen.append(float(loss_gen_e.detach().cpu()))

                    # Plot loss
                    x = np.arange(0, epoch+1)
                    ax.plot(x, loss_disc, label='Discriminator loss', marker='o', color='b')
                    ax.plot(x, loss_gen, label='Generator loss', marker='o', color='r')
                    ax.set_title('Evolution of losses through epochs')
                    ax.set(xlabel='epochs')
                    ax.set(ylabel='loss')
                    if epoch == 0:
                        ax.legend(loc='upper right')
                        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                    # Display current figure
                    ax.grid()
                    # Print progress every epoch
                    print(
                        f"Epoch [{epoch+1}/{num_epochs} - "
                        f"Loss D: {loss_disc_e:.4f}, Loss G: {loss_gen_e:.4f}]"
                        )
                    ax.grid()
                    plt.grid()
                    fig.savefig(os.path.join(config['figures']['dir'],'loss_evol.png'))

        # Save model at the end of all epochs
        if config['models']['save']:
            print("Saving model...")
            save_checkpoint(gen, opt_gen, filename=os.path.join(config['models']['rootdir'], config['models']['gen']))
            save_checkpoint(disc, opt_disc, filename=os.path.join(config['models']['rootdir'], config['models']['disc']))

        # Evaluation of this fold
        if kfold:
            print("\nEvaluation of testing dataset:")
            gen.eval()
            with torch.no_grad():
                # Iterate over the batches of test data and generate super resolution images from downscale of HR test images
                for idx, (low_res, high_res) in enumerate(testloader):
                    # Generate super resolution image from low_res
                    super_res = gen(low_res.to(device))
            # Transfer images to cpu and convert them to arrays
                    super_res = np.asarray(super_res.cpu())
                    high_res = np.asarray(high_res.cpu())
                    # Calculate PSNR
                    # Iterate over all images in this batch 
                    for i, hr_im in enumerate(high_res):
                        psnr[fold, idx, i] = peak_signal_noise_ratio(hr_im, super_res[i])
                        # Calculate SSIM
                        ssim[fold, idx, i] = structural_similarity(hr_im, super_res[i], channel_axis=0)

            # Print averaged PSNR and SSIM
            psnr[fold, :, :][psnr[fold, :, :] == 0] = np.nan
            ssim[fold, :, :][ssim[fold, :, :] == 0] = np.nan
            print(f"Average PSNR of fold {fold+1}/{n_splits}: {np.nanmean(psnr[fold, :, :]):.3f}")
            print(f"Average SSIM of fold {fold+1}/{n_splits}: {np.nanmean(ssim[fold, :, :]):.3f}")
            print(f"----------------------------------------")
            gen.train()

    if kfold:
        print(f"Average PSNR of all folds: {np.nanmean(psnr):.2f}")
        print(f"Average SSIM of all folds: {np.nanmean(ssim):.2f}")
        np.save("psnr.npy", psnr)
        np.save("ssim.npy", ssim)


if __name__ == "__main__":
    train_srgan(learning_rate, num_epochs, batch_size, num_workers)
    print("Finished training.")
