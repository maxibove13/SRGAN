import torch
import os
import config

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

def plot_examples():