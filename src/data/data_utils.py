#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Module with data utilities functions"""

__author__ = "Maximiliano Bove"
__email__ = "maxibove13@gmail.com"
__status__ = "Development"
__date__ = "03/22"

# Built-in modules
import os

# Third-party modules
import numpy as np

from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import yaml


# read config file
with open('config.yaml') as file:
    config = yaml.safe_load(file)

low_res = config['data']['high_res']//config['data']['upscale_factor']

class ImageDataset(Dataset):
    def __init__(self, root_dir):
        super(ImageDataset, self).__init__()
        self.data = []
        self.root_dir = root_dir
        self.class_names = os.listdir(root_dir)
        files = os.listdir(root_dir)
        self.data += list(zip(files, [1] * len(files)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_file, label = self.data[index]
        root_and_dir = self.root_dir

        image = np.array(Image.open(os.path.join(root_and_dir, img_file)))
        if config['data']['dataset'] == 'UxLES':
          image = image[:,:,0:3]
        image = both_transforms(image=image)["image"]
        high_res = highres_transform(image=image)["image"]
        low_res = lowres_transform(image=image)["image"]
        return low_res, high_res


def test():
    dataset = ImageDataset(root_dir="new_data/")
    loader = DataLoader(dataset, batch_size=1, num_workers=8)

    for low_res, high_res in loader:
        print(low_res.shape)
        print(high_res.shape)

# Normalize it and convert it to tensor
highres_transform = A.Compose(
    [
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2(),
    ]
)

# Take a high res image, lower its resolution (24*24) and normalize it
lowres_transform = A.Compose(
    [
        A.Resize(width=low_res, height=low_res, interpolation=Image.BICUBIC),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
        ToTensorV2(),
    ]
)

# Take a high res image and lower its resolution
lr_transform = A.Compose(
    [
        A.Resize(width=low_res, height=low_res, interpolation=Image.BICUBIC),
        ToTensorV2(),
    ]
)

both_transforms = A.Compose(
    [
        A.RandomCrop(width=config['data']['high_res'], height=config['data']['high_res']),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ]
)

test_transform = A.Compose(
    [
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
        ToTensorV2(),
    ]
) 


if __name__ == "__main__":
    test()