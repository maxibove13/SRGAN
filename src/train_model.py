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
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml

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

    # Import High Resolution images (dataset)
    print("Importing dataset...")
    train_dataset = ImageDataset(root_dir=os.path.join(config['data']['rootdir'], config['data']['dataset'], 'HR'))
    print(f"{len(train_dataset)} training samples from {train_dataset.root_dir}")

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

    # Load data
    print(f"Loading dataset with batch size of {batch_size} and {num_workers} workers")
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)

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

    # Training loop
    print(f"SRGAN training: \n")
    print(f" Total training samples: {len(train_dataset)}\n Number of epochs: {num_epochs}\n Mini batch size: {batch_size}\n Number of batches: {len(loader)}\n Learning rate: {learning_rate}\n")

    loss_disc = []
    loss_gen = []

    # Start the stopwatch
    # t0 = process_time()

    fig, ax = plt.subplots(figsize=(10,6), dpi= 80)

    for epoch in range(num_epochs):
        for idx, (low_res, high_res) in enumerate(loader):

            # Send images to device
            high_res = high_res.to(device)
            low_res = low_res.to(device)

            # Generate fake (high_res) image from low_res
            fake = gen(low_res)

            loss_disc_e = train_discriminator(disc, opt_disc, fake, high_res, bce)
            loss_gen_e = train_generator(disc, opt_gen, fake, high_res, vgg_loss_fun, mse)

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
                fig.savefig(os.path.join(config['figures']['dir']/'loss_evol.png'))

        if config['models']['save']:
            save_checkpoint(gen, opt_gen, filename=config['models']['gen'])
            save_checkpoint(disc, opt_disc, filename=config['models']['disc'])

if __name__ == "__main__":
    train_srgan(learning_rate, num_epochs, batch_size, num_workers)