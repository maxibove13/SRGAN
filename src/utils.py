#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Module with general utilities functions"""

__author__ = "Maximiliano Bove"
__email__ = "maxibove13@gmail.com"
__status__ = "Development"
__date__ = "03/22"

# Built-in modules
import os
import random
import time

# Third-party modules
import numpy as np
from matplotlib import rcParams, cm, pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import ImageGrid
from IPython import display
from PIL import Image
import torch
import torchvision.transforms as T
from torchvision.utils import save_image
import yaml

from data.data_utils import ImageDataset


# Adjust font sizes
plt.rc('font', size=10)          # controls default text sizes
plt.rc('axes', titlesize=10)    # fontsize of the axes title
plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14)    # fontsize of the tick labels
plt.rc('legend', fontsize=10)    # legend fontsize
plt.rc('figure', titlesize=10)  # fontsize of the figure title

# Adjust linewidth for axes, ticks, grid, and lines
rcParams['axes.linewidth'] = 1
rcParams['xtick.major.width'] = 1
rcParams['ytick.major.width'] = 1
rcParams['xtick.minor.width'] = 1
rcParams['ytick.minor.width'] = 1
rcParams['grid.linewidth'] = 1
rcParams['lines.linewidth'] = 1
# rcParams['mathtext.fontset'] = 'stix'
# rcParams['font.family'] = 'STIXGeneral'
# rcParams['mathtext.fontset'] = 'custom'
# rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
# rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
# rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'


def unnormalize(mean,std):
    return T.Normalize((-mean / std), (1.0 / std))


WT_D = 126
LIMS = [768,1536,946,1314]

# read config file
with open('config.yaml') as file:
    config = yaml.safe_load(file)

class TestingSet(ImageDataset):
    def __init__(self, root_dir, device):
        super(TestingSet, self).__init__(root_dir, training=False)
        self.device = device
        self.unnorm_lr = unnormalize(0,1)
        self.unnorm_hr = unnormalize(0.5,0.5)
        self.rnd_indices = [random.randrange(0, len(self)-1) for r in range(0,4)]
        self.fig = plt.figure(figsize=(38, 10))
        self.fig_1 = plt.figure(figsize=(70, 10))

    def generate_sr(self, index, gen):
        """generate super resolution image given an index of this dataset"""
        with torch.no_grad():
            lr_image = torch.unsqueeze(self[index][0], 0).to(self.device)
            sr_image = gen(lr_image).cpu()
        return sr_image

    def back_to_original(self, im, image_type):
        """unnormalize and rescale image"""
        # unnormalize them
        if image_type == 'HR':
            im = self.unnorm_hr(torch.unsqueeze(im,0))[0]
        elif image_type == 'LR':
            im = self.unnorm_lr(torch.unsqueeze(im,0))[0]
        # rescale them to the original velocity value
        im = im * (12-2) + 2
        return im

    def plot_grid_of_errors(self, gen):
        # init grid
        grid = ImageGrid(
            self.fig_1, 142,
            nrows_ncols=(1,4),
            axes_pad=(0.5, 0.4),
            share_all=True,
            label_mode="L",
            cbar_mode="edge",
            cbar_location="right"
        )
        
        
        # iterate over grid
        for i, ax in enumerate(grid):
            ax.get_yaxis().set_visible(False)
            # obtain HR image from testing set
            hr_image = self[self.rnd_indices[i]][1][0]
            # unnormalize and rescale
            hr_image = self.back_to_original(hr_image, 'HR')

            # generate synth SR image
            sr_image = self.generate_sr(self.rnd_indices[i], gen)[0,0,:,:]
            # unnormalize and rescale
            sr_image = self.back_to_original(sr_image, 'HR')

            # paint image on ax
            im = ax.imshow(
                sr_image - hr_image, 
                cmap=cm.jet, 
                origin='lower', 
                interpolation='none',
                extent=[l/WT_D for l in LIMS],
                vmin=-1, vmax=1
                )

            # title and labels
            title = 'Error (SR - HR)'
            ax.set_title(f"#{i+1}", pad=6, fontsize=14)
            ax.set_xlabel('$x/D$')
            ax.set_xticks([8, 10, 12])
            if i == 0:
                ax.get_yaxis().set_visible(True)
                ax.set_ylabel('$y/D$')
                ax.set_yticks([8, 9, 10])

        cbar = grid.cbar_axes[0].colorbar(im)
        cbar.set_label('$U_x$ [$ms^{-1}$]', labelpad=18, rotation=270, loc='center')
        # save figure
        self.fig_1.savefig(
            os.path.join('figures', 'image_comparison_err.png'),
            dpi=600,
            bbox_inches='tight'
        )
        plt.clf()

    def plot_grid_of_samples(self, gen):
        """plot a grid of random image samples given a generator"""
        # init grid
        grid = ImageGrid(
            self.fig, 142,
            nrows_ncols=(4,3),
            axes_pad=(0.5, 0.4),
            share_all=True,
            label_mode="L",
        )
        # iterate over grid columns
        for i, ax_col in enumerate(grid.axes_column):
            # iterate over grid rows
            for j, ax in enumerate(ax_col):
                # hide the axis
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

                # if first column show axis
                if i == 0:
                    title = 'Low Resolution (LR)'
                    ax.get_yaxis().set_visible(True)
                    image = self[self.rnd_indices[j]][0][0]
                    image = self.unnorm_lr(torch.unsqueeze(image,0))[0]
                elif i == 1:
                    title = 'Super Resolution (SR)'
                    image = self.generate_sr(self.rnd_indices[j], gen)[0,0,:,:]
                    image = self.unnorm_hr(torch.unsqueeze(image,0))[0]
                elif i == 2:
                    title = 'High Resolution (HR)'
                    image = self[self.rnd_indices[j]][1][0]
                    image = self.unnorm_hr(torch.unsqueeze(image,0))[0]
                if j == 3:
                    ax.get_xaxis().set_visible(True)
                # rescale image to original velocity values
                image = image * (12-2) + 2
                # paint image on axes
                im = ax.imshow(
                    image, 
                    cmap=cm.jet, 
                    origin='lower', 
                    interpolation='none',
                    extent=[l/WT_D for l in LIMS],
                    vmin=2, vmax=12
                    )
                # title and labels
                ax.set_title(f"#{j+1} - {title}", pad=6)
                ax.set_xlabel('$x/D$')
                ax.set_xticks([8, 10, 12])
                ax.set_ylabel('$y/D$')
                ax.set_yticks([8, 9, 10])

        # set colorbar and label
        cbar = self.fig.colorbar(im, ax=grid.cbar_axes[0], orientation='horizontal', location='top', aspect=40)
        cbar.set_label('$U_x$ [$ms^{-1}$]', labelpad=6, rotation=0, loc='center')

        # save figure
        self.fig.savefig(
            os.path.join('figures', 'image_comparison.png'),
            dpi=600,
            bbox_inches='tight'
        )
        plt.clf()

def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    """
    Function to save a checkpoint.
    A net's weights and biases are contain in the model's parameter. A state_dict is simply a dict object that maps each layer to its parameter tensor.
    Here we save the net and optimizer state_dict for later use.
    """
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr, device):
    """
    Function to load a checkpoint.
    """
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # Update lr (otherwise it will have lr of old checkpoint)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def plot_examples(low_res_folder, G, idx):
    """
    Function to plot some generated super_res images examples
    """
    files = [f for f in os.listdir('datasets/testing/') if '.png' in f]

    # Put Generator in evaluation mode
    G.eval()

    for file in files:
        image = Image.open(low_res_folder + file)
        im_arr = np.asarray(image)[:,:,0:3] # Only RGB channels.
        print(f"Opening LR image: {file} - {np.asarray(image)[:,:,0:3].shape}")
        with torch.no_grad():
            upscaled_img = G(
                config.test_transform(image=im_arr)["image"].unsqueeze(0).to(config.DEVICE)
            )
        save_image(upscaled_img * 0.5 + 0.5, f"datasets/testing/sr/{file[:-4]}_sr.png")
        print(f"Saving SR image: {file[:-4]}_sr.png")

    
    # Put generator in training mode
    G.train()


def plot_loss(epoch, loss_disc, loss_gen, dataset, fig, ax, loader):
    x = np.arange(0, epoch+1)
    ax.plot(x, loss_disc, label='Discriminator loss', marker='o', color='b')
    ax.plot(x, loss_gen, label='Generator loss', marker='o', color='r')
    ax.set_title('Evolution of losses through epochs')
    ax.set(xlabel='epochs')
    ax.set(ylabel='loss')
    if epoch == 0:
      ax.legend(loc='upper right')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    display.clear_output(wait=True)
    print(f"SRGAN training: \n")
    print(f" Total training samples: {len(dataset)}\n Number of epochs: {config.NUM_EPOCHS}\n Mini batch size: {config.BATCH_SIZE}\n Number of batches: {len(loader)}\n Learning rate: {config.LEARNING_RATE}\n")
    # Display current figure
    display.display(fig)
    # Pause execution 0.1s
    time.sleep(0.1)