#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Script to test a SRGAN"""

__author__ = "Maximiliano Bove"
__email__ = "maxibove13@gmail.com"
__status__ = "Development"
__date__ = "03/22"


# Built-in modules
import os

# Third party modules
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.utils import save_image
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torch.optim as optim
import yaml

# Local modules
from data.data_utils import ImageDataset, test_transform
from models.model import Generator, Discriminator, VGGLoss
from utils import load_checkpoint
from lower_res import lower_res

# read yaml file
with open('config.yaml') as file:
    config = yaml.safe_load(file)
learning_rate = config['train']['learning_rate']
factor = config['data']['upscale_factor']

def test_srgan(learning_rate, factor):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {torch.cuda.get_device_name()}")

    print('Initializing Generator and optimizer...')
    gen = Generator(in_channels=config['data']['img_channels']).to(device)
    opt_gen = optim.Adam(gen.parameters(), lr=learning_rate, betas=(0.9, 0.999))

    print(f"Loading model: {config['models']['gen']}...")
    load_checkpoint(
            os.path.join(config['models']['rootdir'], config['models']['gen']),
            gen,
            opt_gen,
            learning_rate,
            device
        )
    # Put generator in evaluation mode
    gen.eval()

    hr_dir = os.path.join(
        config['data']['rootdir'], 
        config['data']['dataset'],
        'test')
    # Iterate over all testing images
    for idx, hr_name in enumerate(os.listdir(hr_dir)):
        # Lower the resolution of the testing image
        lower_res(
            hr_dir,
            hr_name,
            factor,
            os.path.join(hr_dir, 'LR')
        )
        # Open LR image
        lr_im = Image.open(os.path.join(hr_dir, 'LR', f'lr_{hr_name}'))
        lr_im_arr = np.asarray(lr_im)[:,:,0:3] # Only RGB channels
        # Pass it to Generator and generate SR image
        with torch.no_grad():
            sr_image = gen(
                test_transform(image=lr_im_arr)["image"].unsqueeze(0).to(device)
                )
        sr_image = sr_image * 0.5 + 0.5
        save_image(sr_image, os.path.join(hr_dir, 'SR', f'sr_{hr_name}'))
        # Open HR image
        print(sr_image.shape)
        hr_image = np.asarray(Image.open(os.path.join(hr_dir, hr_name)))[:, :-2, 0:3]
        sr_image = np.moveaxis(np.asarray(sr_image.cpu())[0,:,:,:], 0, 2)
        # Calculate PSNR and SSIM of SR vs HR images
        print(hr_image.shape)
        print(sr_image.shape)
        psnr = peak_signal_noise_ratio(hr_image, sr_image)
        ssim = structural_similarity(hr_image, sr_image, channel_axis=2)
        print(f"Generated SR image: sr_{hr_name} | PSNR: {psnr:.2f} | SSIM {ssim:.2f}")

    # Make comparison figure
    fig, axs = plt.subplots(3, 3)

    for i in range(3):
        # Open HR image
        hr_image = np.asarray(Image.open(os.listdir(hr_dir)[i]))[:, :, 0:3]
        # Open LR image
        lr_image = np.asarray(Image.open(os.listdir(os.path.join(hr_dir, 'LR'))[i]))[:, :, 0:3]
        # Open SR image
        sr_image = np.asarray(Image.open(os.listdir(os.path.join(hr_dir, 'SR'))[i]))[:, :, 0:3]

        axs[i, 0].imshow(lr_image)
        axs[i, 0].set_title(f"High Resolution - {hr_image.shape}", fontsize=32)
        axs[i, 0].axis('off')
        axs[i, 1].imshow(sr_image)
        axs[i, 1].set_title(f"Super Resolution - {sr_image.shape}", fontsize=32)
        axs[i, 1].imshow(hr_image)
        axs[i, 2].axis('off')
        axs[i, 2].set_title(f"Low Resolution - {lr_image.shape}", fontsize=32)
        axs[i, 2].axis('off')

        plt.savefig(os.path.join('figures', f'lsh_{hr_name}'),  bbox_inches='tight')

if __name__ == "__main__":
    test_srgan(learning_rate, factor)