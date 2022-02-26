#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Script to generate super resolution images from low resolution ones. It uses a pretrained Generator that upsamples the image 4x."""

__author__ = "Maximiliano Bove"
__email__ = "maxibove13@gmail.com"
__status__ = "Development"
__date__ = "02/22"

# Built-in modules
import os
import sys

# Third-party modules
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch.optim as optim

# Local modules
import config
from model import Generator
from utils import load_checkpoint, plot_examples

testing = True # If not testing: validation
r = 3 # Zoom factor

def gen_sr_images(r, testing=True, train_data='DIV2K'):
    """
    Function to generate super resolution images

    Parameters
    ----------
    r : int
        Zoom (crop) factor. Crops the image in the center r times.

    testing : bool = True
        If only testing (no high resolution image to validate. If false looks for the high resolution image to validate the results.

    train_data : str = 'DIV2K'
        Choose which checkpoints to use. That is, either trained by DIV2K dataset or UxLES dataset.

    Returns
    -------
    """
    # Initialize SRGAN Generator
    print("Initializing Generator and optimizer...")
    gen = Generator(in_channels=3).to(config.DEVICE)
    # Define optimizer for Generator
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.9, 0.999))

    print("Loading checkpoint...")
    # Load checkpoint (w&b of specified training)
    if config.LOAD_MODEL:
        load_checkpoint(
            f'checkpoints/{train_data}/{config.CHECKPOINT_GEN}',
            gen,
            opt_gen,
            config.LEARNING_RATE
        )

    print("Generating Super Resolution images...")
    # Test generator in all images in testing folder
    plot_examples("datasets/testing/", gen, 0)

    print("Preparing comparison figure...")
    # Get list of generated super resolution images
    sr_images = next(os.walk("datasets/testing/sr/"))[2]

    # Get list of testing images
    test_images = next(os.walk("datasets/testing/"))[2]

    # If validating, get hr images
    if not testing:
        hr_images = next(os.walk("new_data/hr/"))[2:][0]

    # Loop through all test_images
    for idx, im in enumerate(test_images):
        # Read lr, sr and hr images
        lr_im = mpimg.imread(f"datasets/testing/{im}")
        sr_im = mpimg.imread(f"datasets/testing/sr/{sr_images[idx]}")
        if not testing:
            hr_im = mpimg.imread(f"new_data/hr/{hr_images[idx]}")

        # Get new widths and heights given zoom in factor r
        w0 = sr_im.shape[0]//2-sr_im.shape[0]//r
        w1 = sr_im.shape[0]//2+sr_im.shape[0]//r
        h1 = sr_im.shape[1]//2-sr_im.shape[1]//r
        h2 = sr_im.shape[1]//2+sr_im.shape[1]//r
        w0_l = lr_im.shape[0]//2-lr_im.shape[0]//r
        w1_l = lr_im.shape[0]//2+lr_im.shape[0]//r
        h1_l = lr_im.shape[1]//2-lr_im.shape[1]//r
        h2_l = lr_im.shape[1]//2+lr_im.shape[1]//r

        # Crop images and define titles for comparison figure
        ims = [lr_im[w0_l:w1_l, h1_l:h2_l, :], sr_im[w0:w1, h1:h2, :]]
        titles = ["Low Resolution", "Super Resolution"]
        if not testing:
            ims.append(hr_im[w0:w1, h1:h2, :])
            titles.append('High Resolution')

        # Initialize figure and axes
        fig, axs = plt.subplots(1,2) if testing else plt.subplots(1,3)
        
        # show image in each subplot, set title deactivate axis
        for idx, ax in enumerate(ims):
            axs[idx].imshow(ax)
            axs[idx].set_title(titles[idx], fontsize=36)
            axs[idx].axis('off')

        # Set size and save figure
        fig.set_size_inches((40, 40), forward=False)
        fig.savefig(f"figures/test_{im}.png", bbox_inches='tight')

        print("Test finished.")

if __name__ == "__main__":
    r = sys.argv[1]
    testing = True
    train_data='DIV2K'
    if len(sys.argv) > 2:
        train_data = str(sys.argv[3])
        if len(sys.argv) > 3:
            if sys.argv[2] == 'validation':
               testing = False
    gen_sr_images(r, testing=testing, train_data=train_data)