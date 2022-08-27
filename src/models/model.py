#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Module containing SRGAN building blocks (Convolutional, Upsample and residual), Generator and Discriminator classes"""

__author__ = "Maximiliano Bove"
__email__ = "maxibove13@gmail.com"
__status__ = "Development"
__date__ = "02/22"

# Built-in modules
import os
# Third-party modules
import torch
from torch import nn, tanh
from torchvision.models import vgg19
import yaml
# Local modules

# Define device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# read yaml file
with open('config.yaml') as file:
    config = yaml.safe_load(file)

class ConvBlock(nn.Module):
    # Conv -> BN (Batch Norm) -> Leaky/PReLu
    def __init__(
        self,
        in_channels,
        out_channels,
        discriminator=False,
        use_act=True,
        use_bn=True,
        **kwargs,
    ):
        super().__init__()
        self.use_act = use_act
        self.cnn = nn.Conv2d(in_channels, out_channels, **kwargs, bias=not use_bn)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.act = (
            nn.LeakyReLU(0.2, inplace=True) if discriminator else nn.PReLU(num_parameters=out_channels)
        )
    
    def forward(self, x):
        return self.act(self.bn(self.cnn(x))) if self.use_act else self.bn(self.cnn(x))


class UpsampleBlock(nn.Module):
    def __init__(self, in_c, scale_factor):
        super().__init__()
        self.conv = nn.Conv2d(in_c, in_c * scale_factor ** 2, 3, 1, 1)
        self.ps = nn.PixelShuffle(scale_factor) # in_c *4, H, W, --> in_c, H*2, W*2
        self.act = nn.PReLU(num_parameters=in_c)

    def forward(self, x):
        return self.act(self.ps(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.block1 = ConvBlock(
            in_channels,
            in_channels,
            kernel_size = 3,
            stride = 1,
            padding = 1
        )
        self.block2 = ConvBlock(
            in_channels,
            in_channels,
            kernel_size = 3,
            stride = 1,
            padding = 1,
            use_act = False
        )

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        return out + x


class Generator(nn.Module):
    def __init__(self, in_channels=3, num_channels=64, num_blocks=16):
        super().__init__()
        self.initial = ConvBlock(in_channels, num_channels, kernel_size=9, stride=1, padding=4, use_bn=False)
        self.residuals = nn.Sequential(*[ResidualBlock(num_channels) for _ in range(num_blocks)])
        self.convblock = ConvBlock(num_channels, num_channels, kernel_size=3, stride=1, padding=1, use_act=False)
        self.upsamples = nn.Sequential(UpsampleBlock(num_channels, 2), UpsampleBlock(num_channels, 2))
        self.final = nn.Conv2d(num_channels, in_channels, kernel_size=9, stride=1, padding=4)

    def forward(self, x):
        initial = self.initial(x)
        x = self.residuals(initial)
        x = self.convblock(x) + initial
        x = self.upsamples(x)
        return tanh(self.final(x))


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 64, 128, 128, 256, 256, 512, 512]):
        super().__init__()
        blocks = []
        for idx, feature in enumerate(features):
            blocks.append(
                ConvBlock(
                    in_channels,
                    feature,
                    kernel_size = 3,
                    stride = 1 + idx % 2,
                    padding = 1,
                    discriminator = True,
                    use_act = True,
                    use_bn = False if idx == 0 else True
                )
            )
            in_channels = feature

        self.blocks = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((6,6)),
            nn.Flatten(),
            nn.Linear(512*6*6, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1)
        )

    def forward(self, x):
        x = self.blocks(x)
        return self.classifier(x)
        
class VGGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        if not config['models']['dwnld_vgg']:
            self.vgg = vgg19(pretrained=False)
            pretrained_vgg = torch.load(os.path.join(config['models']['rootdir'], 'vgg19-dcbb9e9d.pth'))
            self.vgg.load_state_dict(pretrained_vgg)
            self.vgg = self.vgg.features[:36].eval().to(device)
        else:
            self.vgg = vgg19(weights='IMAGENET1K_V1').features[:36].eval().to(device)
        self.loss = nn.MSELoss()

        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, input, target):
        vgg_input_features = self.vgg(input)
        vgg_target_features = self.vgg(target)
        return self.loss(vgg_input_features, vgg_target_features)

### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
def train_discriminator(D, opt, fake, high_res, bce):

    # Reset gradients to zero
    opt.zero_grad()

    # Train on real data
    pred_real = D(high_res)
    loss_real = bce(pred_real, torch.ones_like(pred_real) - 0.1 * torch.rand_like(pred_real))

    # Train on fake data
    pred_fake = D(fake.detach())
    loss_fake = bce(pred_fake, torch.zeros_like(pred_fake))

    loss = (loss_real + loss_fake)

    # Backward pass
    loss.backward()
    # Update weights
    opt.step()

    return loss_real, loss_fake

### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
def train_generator(D, opt, fake, high_res, vgg_loss, mse, bce):

    # Reset gradients to zero
    opt.zero_grad()

    pred_fake = D(fake)

    adv_loss = 1e-3 * bce(pred_fake, torch.ones_like(pred_fake))
    vgg_loss = 0.006 * vgg_loss(fake, high_res)
    mse_loss = mse(fake, high_res)

    loss = adv_loss + vgg_loss + mse_loss

    # Backward pass
    loss.backward()
    # Update weights
    opt.step()

    return adv_loss, vgg_loss, mse_loss