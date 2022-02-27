import numpy as np
import torch
import os
import config
from PIL import Image
from torchvision.utils import save_image
import matplotlib.pyplot as plt

def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    """
    Function to save a checkpoint.
    A net's weights and biases are contain in the model's parameter. A state_dict is simply a dict object that maps each layer to its parameter tensor.
    Here we save the net and optimizer state_dict for later use.
    """
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    """
    Function to load a checkpoint.
    """
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # Update lr (otherwise it will have lr of old checkpoint)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def plot_examples(low_res_folder, G, idx):
    """
    Function to plot some generated super_res images examples
    """
    files = [f for f in os.listdir('datasets/testing/') if '.png' in f]

    # Put Generator in evaluation mode
    G.eval()

    for file in files:
        image = Image.open(low_res_folder + file)
        im_arr = np.asarray(image)[:,:,0:3] # Only RGB channels.
        print(f"Opening LR image: {file} - {np.asarray(image)[:,:,0:3].shape}")
        with torch.no_grad():
            upscaled_img = G(
                config.test_transform(image=im_arr)["image"].unsqueeze(0).to(config.DEVICE)
            )
        save_image(upscaled_img * 0.5 + 0.5, f"datasets/testing/sr/{file[:-4]}_sr.png")
        print(f"Saving SR image: {file[:-4]}_sr.png")

    
    # Put generator in training mode
    G.train()

def plot_loss(epoch, loss_disc, loss_gen):
  x = np.arange(0, epoch)
  plt.plot(x, loss_disc, label='Discriminator loss')
  plt.plot(x, loss_gen, label='Generator loss')

  plt.title('Evolution of losses through epochs')
  plt.xlabel('epochs')
  plt.ylabel('loss')
  plt.legend(loc='upper right')
  plt.grid()