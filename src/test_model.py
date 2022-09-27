#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Script to test a SRGAN"""

__author__ = "Maximiliano Bove"
__email__ = "maxibove13@gmail.com"
__status__ = "Development"
__date__ = "03/22"


# Built-in modules
import os
import time

# Third party modules
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image
from torchvision.utils import save_image
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torch.optim as optim
import yaml

# Local modules
from data.data_utils import test_transform
from models.model import Generator
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
        'test',
        'dataset')
    lr_dir = os.path.join(hr_dir, 'LR')
    sr_dir = os.path.join(hr_dir, 'SR')
    hr_list = [hr for hr in os.listdir(hr_dir) if hr[-4:] == '.png']
    # Iterate over all testing images
    print("Generating SR images:")
    psnr = np.zeros((len(hr_list)))
    ssim = np.zeros((len(hr_list)))
    for idx, hr_name in enumerate(hr_list):
        # print(idx)
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
        # Open HR and SR images for comparison
        hr_image = Image.open(os.path.join(hr_dir, hr_name))
        sr_image = Image.open(os.path.join(sr_dir, 'sr_' + hr_name))
        # Resize HR image to SR size (it may be different by a few pixels)
        hr_image = hr_image.resize(sr_image.size)

        # Calculate PSNR and SSIM of SR vs HR images
        psnr[idx] = peak_signal_noise_ratio(np.array(hr_image)[:, :, 0:3], np.array(sr_image))
        ssim[idx] = structural_similarity(np.array(hr_image)[:, :, 0:3], np.array(sr_image), channel_axis=2)
        # print(f"sr_{hr_name} | PSNR: {psnr[idx]:.2f} | SSIM {ssim[idx]:.2f}")

    # Make comparison figure
    print("Creating comparison figure...")
    fig, axs = plt.subplots(4, 4)

    images = [6, 8, 9, 1]

    # for idx, i in enumerate(images):
    #     # Open HR image
    #     hr_path = os.path.join(hr_dir, hr_list[i])
    #     hr_image = np.asarray(Image.open(hr_path))[:, :, 0:3]
    #     # Open LR image
    #     lr_name = [lr for lr in os.listdir(lr_dir) if lr[3:] == hr_list[i]][0]
    #     lr_path = os.path.join(lr_dir, lr_name)
    #     lr_image = np.asarray(Image.open(lr_path))[:, :, 0:3]
    #     # Open SR image
    #     sr_name = [sr for sr in os.listdir(sr_dir) if sr[3:] == hr_list[i]][0]
    #     sr_path = os.path.join(sr_dir, sr_name)
    #     sr_image = np.asarray(Image.open(sr_path))[:, :, 0:3]

        

    #     print(lr_image.shape)
    #     im_lr = axs[0, idx].imshow(lr_image)
    #     axs[0, idx].set_title(f"Low Resolution - {lr_image.shape}", fontsize=16)
    #     # axs[0, idx].axis('off')
    #     if idx == len(images)-1:
    #         divider = make_axes_locatable(axs[0, idx])
    #         cax = divider.append_axes('right', size='5%', pad=0.05)
    #         fig.colorbar(im_lr, cax=cax, orientation='vertical')

    #     axs[1, idx].imshow(sr_image)
    #     axs[1, idx].set_title(f"Super Resolution - {sr_image.shape}", fontsize=16)
    #     # axs[1, idx].axis('off')

    #     axs[2, idx].imshow(hr_image)
    #     axs[2, idx].set_title(f"High Resolution - {hr_image.shape}", fontsize=16)
    #     # axs[2, idx].axis('off')

    #     axs[3, idx].imshow(sr_image-hr_image)
    #     axs[3, idx].set_title(f"Error - {hr_image.shape}", fontsize=16)
    #     # axs[3, idx].axis('off')

    #     plt.subplots_adjust(wspace=0.1, hspace=0.07)
    #     fig.set_size_inches((20, 30), forward=False)
    #     plt.savefig(os.path.join('figures', f'lsh_{hr_name}'),  bbox_inches='tight')

    # Create histogram

    # fig_h, axs_h = plt.subplots(1,2)
    # axs_h[0].hist(psnr, 20, density=True, facecolor='C0', edgecolor='black', label='PSNR')
    # axs_h[0].set_xlabel('dB')
    # axs_h[0].set_ylabel('Probability density')
    # axs_h[0].set_title(f'$\mu={np.mean(psnr):.2f}$, $\sigma={np.std(psnr):.2f}$')
   
    # axs_h[1].hist(ssim, 20, density=True, facecolor='C1', edgecolor='black', label='SSIM')
    # axs_h[1].set_title(f'$\mu={np.mean(ssim):.2f}$, $\sigma={np.std(ssim):.2f}$')

    # axs_h[0].legend()
    # axs_h[1].legend()
   
    # fig_h.savefig(os.path.join('figures', f'histogram'),  bbox_inches='tight') 

    # print(f"Average PSNR and SSIM of all testing images:")
    # print(f"PSNR: Mean: {np.mean(psnr):.2f} Std: {np.std(psnr):.2f}")
    # print(f"SSIM: Mean: {np.mean(ssim):.2f} Std: {np.std(ssim):.2f}")

if __name__ == "__main__":
    tic = time.time()
    test_srgan(learning_rate, factor)
    toc = time.time()

    print(toc-tic)