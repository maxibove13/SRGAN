#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Script to lower the resolution of an image and generate a figure comparing the HR and LR version."""

__author__ = "Maximiliano Bove"
__email__ = "maxibove13@gmail.com"
__status__ = "Development"
__date__ = "02/22"

import os
import sys

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from data.data_utils import lowres_transform
import yaml

# read yaml file
with open('config.yaml') as file:
    config = yaml.safe_load(file)
factor = config['data']['upscale_factor']

def lower_res(hr_dir, hr_name, factor, lr_dir, hr_lr_fig=False, prints=False):
  # Open HR image
  hr_image = np.array(Image.open(os.path.join(hr_dir, hr_name)))[:, :, 0:3]
  # Downscale it
  if prints:
    print(f'HR image size: {hr_image.shape}')
  lr_image = A.Compose(
    [
      A.Resize(width=hr_image.shape[1]//factor, height=hr_image.shape[0]//factor),
      A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
      ToTensorV2()
    ]
  )(image=hr_image)["image"]
  # Save it
  lr_image = lr_image * 0.5 + 0.5
  save_image(lr_image, f'{lr_dir}/lr_{hr_name}')
  lr_image = np.moveaxis(np.asarray(lr_image), 0, 2)
  
  if prints:
    print(f"Lowered {factor}x the resolution of {hr_name} image located in {hr_dir} and saved it in {lr_dir}")
    print(f'LR image size: {lr_image.shape}')
  if hr_lr_fig:
    # Generate figure comparing both image
    fig, axs = plt.subplots(1,2)
    axs[1].imshow(hr_image)
    axs[1].set_title(f"High Resolution - {hr_image.shape}", fontsize=32)
    axs[1].axis('off')
    axs[0].imshow(lr_image)
    axs[0].set_title(f"Low Resolution - {lr_image.shape}", fontsize=32)
    axs[0].axis('off')


    fig.set_size_inches((20, 10), forward=False)
    plt.savefig(os.path.join('figures', f'hr_lr_{hr_name}'),  bbox_inches='tight')

if __name__ == "__main__":
  hr_dir = 'data/UxLES'
  lr_dir = 'data/lowered'
  hr_name = 'hr_91960_296.png'
  lower_res(hr_dir, hr_name, factor, lr_dir, hr_lr_fig=True, prints=True)
