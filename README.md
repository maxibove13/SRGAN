# Super Resolution GAN

A naive implementation of a Super-Resolution GAN.
Based on the work by Ledig et. al. 2017
arxiv.org/abs/1609.04802

![Example of SRGAN using DIV2K dataset](https://github.com/maxibove13/SRGAN/blob/main/figures/div2k_example.png?raw=true)
![Example of SRGAN using UxLES dataset](https://github.com/maxibove13/SRGAN/blob/main/figures/uxles_example_example.png?raw=true)

## Instructions

1. Copy `config_sample.yaml` file and rename it `config.yaml` in order to modify any configuration parameter you want without modifying the version control.

2. Download either `DIV2K` or `UxLES` dataset and extract them in `data/`

   - [DIV2K dataset](https://drive.google.com/file/d/1OHo_hmFTqAkgpjo6sNf_1iuWeDsDad0T/view?usp=sharing)
   - [UxLES dataset](https://drive.google.com/file/d/1Khhfgz9_Di7S6PZFs5tmK_qRT-Y-jbNH/view?usp=sharing)

3. Optionally, download the pretrained Generators and Discriminators:

   - [DIV2K Generator](https://drive.google.com/file/d/1xK8VOXJ--SCAvlY32SzUph8S1A2bxG08/view?usp=sharing)
   - [DIV2K Discriminator](https://drive.google.com/file/d/1hr1e6E0GCy7IIkAUweChzPPY07s-p6Rv/view?usp=sharing)

   - [UxLES Generator](https://drive.google.com/file/d/1v6TqUhTkZ8WYfsr4ZNb8GF5eT_l1Dy3w/view?usp=sharing)
   - [UxLES Discriminator](https://drive.google.com/file/d/1NI3pDJ4VxtegQsFYQCGQaJVWMsGYdjWq/view?usp=sharing)

Now you should be ready to use the repository.

## config.yaml

This is the configuration file, it is divided in:

### data

Here you should specify the root directory of the `data` samples, the `dataset` (either `DIV2K`, `UxLES` or another of your choice), the `high_res` size, that is the crop of high resolution images that are going to feed the training network (recomended 96 in `DIV2K` and 64 in `UxLES`), the `upscale_factor` (4 by default) and the `img_channels` that should be 3 (RGB) in most cases.

### models

You can choose to download the pretrained VGG network that is used for one of the Generator loss terms from `torch` if you set `dwnld_vgg` to False, or set it to True and download it previously from here:
[Pretrained VGG19](https://drive.google.com/file/d/1xK8VOXJ--SCAvlY32SzUph8S1A2bxG08/view?usp=sharing)

You should also specify if you want to load a pretrained Generator and Discriminator during training and if to save it at the end of each epoch. Also the filename and directory of the model to load or save must be stated.

### train

Here the usual hyperparameters are specified: `learning_rate`, `num_epochs`, `batch_size` and `num_workers`.

### validation

If to apply kfold cross validation during training or not, and how many splits.

### figures

Just state the directory where you want to save the figures that result from training or testing.

## Train the network

Setup the training session and the hyperparameters in `config.yaml`. There, you should check that the data root directory and dataset are the ones you want to train with. If you are training with `UxLES` check that `high_res` is 64 or less.

Run  `train_model.py` script or `run.sh &` to run the training in the background and dump the prints in a `train.log` file.

```
python3 ./src/train_model.py
```

## Test the network

In order to test the network on the testing dataset, make sure you extracted some dataset into `data` directory.

The script will lower the resolution of all testing images and evaluate the Generator on them. It will also generate a comparison figure, and create a PSNR and SSIM histogram.

Run  `train_model.py` script

```
python3 ./src/train_model.py
```


## Google Colab

If you want to run this network in `google colab`:

After you clone the repository in a `Google Drive` folder, mount it to your colab session in order to use the repository features.

```
from google.colab import drive
drive.mount('/content/drive')

# Navigate to repository
%cd /content/drive/MyDrive/<Path_to_repo>

# For some reason we need to explicitly install this package
!pip install albumentations==0.4.6
```

## K-fold

K-fold is a technique to evaluate a model performance.

By default, we split the data into two parts, training and testing. Then we check if the model trained with training data performs well on testing data. The issue with this is that there is a risk of overfitting on the test set as it is used to tweak the hyperparamaters. This way, knowledge about the test set can leak into the model and evaluation metrics no longer report on generalization performance. To solve this problem, yet another part of the dataset can be held out as a so-called 'validation set'. Training proceeds on the training set, after which evaluation is done on the validation set, and when te experiment seems to be succesful, final evaluation can be done on the test set.

However, by partitioning the available data into three sets, we drastically reduce the number of samples which can be used for learning the model, and the results can depend on a particular random choice for the pair of (train, validation) sets.

A solution to this problem is a procedure called cross-validation (CV for short). A test set should still be held out for final evaluation, but the validation set is no longer needed when doing CV. In the basic approach, called k-fold CV, the training set is split into k smaller sets. The following procedure is followed for each of the k “folds”:

   - A model is trained using k-1 of the folds as training data;

   - the resulting model is validated on the remaining part of the data (i.e., it is used as a test set to compute a performance measure such as accuracy).

The performance measure reported by k-fold cross-validation is then the average of the values computed in the loop. This approach can be computationally expensive, but does not waste too much data (as is the case when fixing an arbitrary validation set).