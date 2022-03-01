#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Python script to lower the resolution of all images in HR folder of the 
selected mode (training, validation or testing) and selected dataset."""

__author__ = "Maximiliano Bove"
__email__ = "maxibove13@gmail.com"
__status__ = "Development"
__date__ = "02/22"

import os
import sys

import numpy as np
from PIL import Image
from torchvision.utils import save_image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import lowres_transform

def lower_res(mode, dataset, f):
  """
  Function to generate super resolution images

  Parameters
  ----------

  mode : str
      training, validation or testing mode

  dataset : str
    Training dataset, 'DIV2K' or 'UxLES'

  f : int
    Factor of downscaling.
      

  Returns
  -------
  """
  hr_dir = f"datasets/{mode}/{dataset}/HR"
  hr_images = os.listdir(hr_dir)

  for i in hr_images:
      # Open HR image
      hr_image = np.array(Image.open(os.path.join(hr_dir,i)))[:,:,0:3]
      # Downscale it
      lr_image = A.Compose(
          [
          A.Resize(width=hr_image.shape[1]//f, height=hr_image.shape[0]//f),
          A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
          ToTensorV2()
          ]
      )(image=hr_image)["image"]
      # Save it
      save_image(lr_image, f"datasets/{mode}/{dataset}/LR/{i[:-4]}_lr.png")
    
  print(f"Lowered {f} times the resolution of {len(hr_images)} images in 'datasets/{mode}/{dataset}/HR'\nSaved them in 'datasets/{mode}/{dataset}/LR'")

if __name__ == "__main__":
  mode = sys.argv[1]
  dataset = sys.argv[2]
  f = int(sys.argv[3])
  lower_res(mode, dataset, f)
