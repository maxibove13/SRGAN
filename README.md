# Super Resolution GAN

A naive implementation of a Super-Resolution GAN.
Based on the work by Ledig et. al. 2017
arxiv.org/abs/1609.04802

## Train the network

Run cells in `srgan.ipynb`

## Validate the network
## Test the network

In order to use the network to generate super resolution images from low resolution ones you can load a checkpoint with the last training and evaluate the Generator:

```bash
python test_gen.py <r> <train_data>
```

Where `r` is a cropping factor for comparison purposes and `train_data` is the dataset that the SRGAN was trained with.
Two options available: `DIV2K` or `UxLES`