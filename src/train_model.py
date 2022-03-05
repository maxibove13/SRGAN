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
    dataset = ImageDataset(root_dir=os.path.join(config['data']['rootdir'], config['data']['dataset'], 'HR'))
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

    n_splits = config['validation']['n_splits']
    kfold = KFold(n_splits=n_splits, shuffle=True)

    # Loop through different folds
    psnr = np.zeros((len(dataset)//n_splits, n_splits))
    ssim = np.zeros((len(dataset)//n_splits, n_splits))
    for fold, (train_idx, test_idx) in enumerate(kfold.split(dataset)):
 
        train_subsampler = SubsetRandomSampler(train_idx)
        test_subsampler = SubsetRandomSampler(test_idx)
        
        # Load data
        print(f"Loading training dataset with batch size of {batch_size} and {num_workers} workers ")
        
        trainloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True, num_workers=num_workers, sampler=train_subsampler)
        testloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True, num_workers=num_workers, sampler=test_subsampler)

        # Training loop
        print(f"SRGAN training: \n")
        print(f"k-fold: {fold}/{n_splits} \nTraining samples on this fold: {len(train_subsampler)}\n Number of epochs: {num_epochs}\n Mini batch size: {batch_size}\n Number of batches: {len(trainloader)}\n Learning rate: {learning_rate}\n")

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
                        f"Epoch [{epoch}/{num_epochs} - "
                        f"Loss D: {loss_disc_e:.4f}, Loss G: {loss_gen_e:.4f}]"
                        )
                    ax.grid()
                    plt.grid()
                    fig.savefig(os.path.join(config['figures']['dir'],'loss_evol.png'))

            if config['models']['save']:
                save_checkpoint(gen, opt_gen, filename=config['models']['gen'])
                save_checkpoint(disc, opt_disc, filename=config['models']['disc'])

        print(f"Training process of fold {fold}/{n_splits} has finished.")

        # Evaluation of this fold
        gen.eval()
        with torch.no_grad():
            # Iterate over the test data and generate super resolution images from downscale of HR test images
            psnr = np.zeros(len(testloader))
            for idx, (low_res, high_res) in enumerate(testloader):
                # Generate super resolution image from low_res
                super_res = gen(low_res.to(device))
                # Calculate PSNR
                psnr[idx, fold] = peak_signal_noise_ratio(high_res, super_res)
                # Calculate SSIM
                ssim[idx, fold] = structural_similarity(high_res, super_res)
        # Print averaged PSNR and SSIM
        print(f"Average PSNR of fold {fold}/{n_splits}: {np.mean(psnr[:, fold])}")
        print(f"Average SSIM of fold {fold}/{n_splits}: {np.mean(ssim[:, fold])}")
        gen.train()


if __name__ == "__main__":
    train_srgan(learning_rate, num_epochs, batch_size, num_workers)