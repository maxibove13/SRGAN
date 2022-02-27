# Super Resolution GAN

A naive implementation of a Super-Resolution GAN.
Based on the work by Ledig et. al. 2017
arxiv.org/abs/1609.04802

## Train the network

Run cells in `srgan.ipynb`

## Validate the network
## Test the network

In order to use the network to generate super resolution images from low resolution ones you can load a checkpoint with the last training given the training dataset and evaluate the Generator:

Download this two [folders](https://drive.google.com/drive/folders/11Q37jVKt41J3y72ifBImR1suknSubcVN?usp=sharing) containing the checkpoints inside `checkpoints/` repository directory.

```bash
python test_gen.py <r> <train_data>
```

Where `r` is a cropping factor for comparison purposes and `train_data` is the dataset that the SRGAN was trained with.
Two options available: `DIV2K` or `UxLES`

## Google Colab

After you clone the repository in a `Google Drive` folder, mount it to your colab session in order to use the repository features.

```
from google.colab import drive
drive.mount('/content/drive')

# Navigate to repository
%cd /content/drive/MyDrive/<Path_to_repo>

# For some reason we need to explicitly install this package
!pip install albumentations==0.4.6
```