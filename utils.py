import numpy as np
import torch
import os
import config
from PIL import Image
from torchvision.utils import save_image

def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    """
    Function to save a checkpoint.
    A net's weights and biases are contain in the model's parameter. A state_dict is simply a dict object that maps each layer to its parameter tensor.
    Here we save the net and optimizer state_dict for later use.
    """
    print("=> Saving checkpoint")
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


def plot_examples(low_res_folder, G):
    """
    Function to plot some generated high_res images examples
    """
    files = os.listdir(low_res_folder)

    # Put Generator in evaluation mode
    G.eval()

    for file in files:
        image = Image.open("test_images/" + file)
        with torch.no_grad():
            upscaled_img = G(
                config.test_transform(image=np.asarray(image))["image"].unsqueeze(0).to(config.DEVICE)
            )
        save_image(upscaled_img * 0.5 + 0.5, f"saved/{file}")
    
    # Put generator in training mode
    G.train()