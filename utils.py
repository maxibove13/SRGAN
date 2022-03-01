import os
import time
from time import process_time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from IPython import display
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image

import config


class MyImageFolder(Dataset):
    def __init__(self, root_dir):
        super(MyImageFolder, self).__init__()
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
        if config.TRAINING_SET == 'UxLES':
          image = image[:,:,0:3]
        image = config.both_transforms(image=image)["image"]
        high_res = config.highres_transform(image=image)["image"]
        low_res = config.lowres_transform(image=image)["image"]
        return low_res, high_res

def test():
    dataset = MyImageFolder(root_dir="new_data/")
    loader = DataLoader(dataset, batch_size=1, num_workers=8)

    for low_res, high_res in loader:
        print(low_res.shape)
        print(high_res.shape)

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


def plot_loss(epoch, loss_disc, loss_gen, dataset, fig, ax, loader):
    x = np.arange(0, epoch+1)
    ax.plot(x, loss_disc, label='Discriminator loss', marker='o', color='b')
    ax.plot(x, loss_gen, label='Generator loss', marker='o', color='r')
    ax.set_title('Evolution of losses through epochs')
    ax.set(xlabel='epochs')
    ax.set(ylabel='loss')
    if epoch == 0:
      ax.legend(loc='upper right')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    display.clear_output(wait=True)
    print(f"SRGAN training: \n")
    print(f" Total training samples: {len(dataset)}\n Number of epochs: {config.NUM_EPOCHS}\n Mini batch size: {config.BATCH_SIZE}\n Number of batches: {len(loader)}\n Learning rate: {config.LEARNING_RATE}\n")
    # Display current figure
    display.display(fig)
    # Pause execution 0.1s
    time.sleep(0.1)

