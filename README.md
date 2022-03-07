# Super Resolution GAN

A naive implementation of a Super-Resolution GAN.
Based on the work by Ledig et. al. 2017
arxiv.org/abs/1609.04802

## Instructions

Copy `config_sample.yaml` file and rename it `config.yaml` in order to modify any configuration parameter you want without modifying the version control.

## Train the network

Run  `train_model.py` script

```
python3 ./src/train_model.py
```

## Test the network

Run  `train_model.py` script

```
python3 ./src/train_model.py
```


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

## K-fold

K-fold is a technique to evaluate a model performance.

By default, we split the data into two parts, training and testing. Then we check if the model trained with training data performs well on testing data. The issue with this is that there is a risk of overfitting on the test set as it is used to tweak the hyperparamaters. This way, knowledge about the test set can leak into the model and evaluation metrics no longer report on generalization performance. To solve this problem, yet another part of the dataset can be held out as a so-called 'validation set'. Training proceeds on the training set, after which evaluation is done on the validation set, and when te experiment seems to be succesful, final evaluation can be done on the test set.

However, by partitioning the available data into three sets, we drastically reduce the number of samples which can be used for learning the model, and the results can depend on a particular random choice for the pair of (train, validation) sets.

A solution to this problem is a procedure called cross-validation (CV for short). A test set should still be held out for final evaluation, but the validation set is no longer needed when doing CV. In the basic approach, called k-fold CV, the training set is split into k smaller sets. The following procedure is followed for each of the k “folds”:

   - A model is trained using k-1 of the folds as training data;

   - the resulting model is validated on the remaining part of the data (i.e., it is used as a test set to compute a performance measure such as accuracy).

The performance measure reported by k-fold cross-validation is then the average of the values computed in the loop. This approach can be computationally expensive, but does not waste too much data (as is the case when fixing an arbitrary validation set).