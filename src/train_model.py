#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Script to train a SRGAN"""

__author__ = "Maximiliano Bove"
__email__ = "maxibove13@gmail.com"
__status__ = "Development"
__date__ = "02/22"


# Built-in modules
import os

# Third party modules
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from sklearn.model_selection import KFold
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
import yaml
# from tqdm import tqdm

# Local modules
from data.data_utils import ImageDataset
from models.model import Generator, Discriminator, VGGLoss, train_discriminator, train_generator
from utils import load_checkpoint, save_checkpoint

# read yaml file
with open('config.yaml') as file:
    config = yaml.safe_load(file)
# By default hyperparameters are defined by config.yaml
learning_rate = config['train']['learning_rate']
num_epochs = config['train']['num_epochs']
batch_size = config['train']['batch_size']
num_workers = config['train']['num_workers']

def train_srgan(learning_rate, num_epochs, batch_size, num_workers):

    # Print test comparison figure between SR and HR.
    test_sr = True

    # Define device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {torch.cuda.get_device_name()}")

    # Import High Resolution images (dataset)
    print("Importing dataset...")
    dataset = ImageDataset(root_dir=os.path.join(config['data']['rootdir'], config['data']['dataset'], 'train'))
    print(f"{len(dataset)} training samples from {dataset.root_dir}")



    # Initialize models and send them to device
    print('Initializing Generator and Discriminator...')
    gen = Generator(in_channels=config['data']['img_channels']).to(device)
    disc = Discriminator(in_channels=config['data']['img_channels']).to(device)

    print('Defining losses and optimizers...')
    # Define losses
    bce = nn.BCEWithLogitsLoss()
    vgg_loss_fun = VGGLoss()
    mse = nn.MSELoss()

    # Define optimizers
    opt_gen = optim.Adam(gen.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=learning_rate, betas=(0.9, 0.999))

    # Load checkpoint
    if config['models']['load']:
        print(f"Loading pretrained model from {config['models']['rootdir']}")
        load_checkpoint(
            os.path.join(config['models']['rootdir'], config['models']['gen']),
            gen,
            opt_gen,
            learning_rate,
            device
        )
        load_checkpoint(
            os.path.join(config['models']['rootdir'], config['models']['disc']), disc, opt_disc, learning_rate, 
            device
        )

    kfold = config['validation']['kfold']
    if kfold:
        n_splits = config['validation']['n_splits']
        kfold = KFold(n_splits=n_splits, shuffle=True)
        psnr = np.zeros((n_splits, len(dataset)//n_splits//batch_size + 1, batch_size))
        ssim = np.zeros((n_splits, len(dataset)//n_splits//batch_size + 1, batch_size))

    # Iterate over kfolds only if kfold is on.
    for fold, (train_idx, test_idx) in enumerate(kfold.split(dataset)) if kfold else enumerate([(0,0)]):

        if kfold:
            train_subsampler = SubsetRandomSampler(train_idx)
            test_subsampler = SubsetRandomSampler(test_idx)
            # Load data
            print(f"\nSpliting and loading datasets [{fold+1}/{n_splits}]...")
            trainloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True, num_workers=num_workers, sampler=train_subsampler)
            testloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True, num_workers=num_workers, sampler=test_subsampler)
        else:
            print(f"\nLoading dataset...")
            trainloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True, num_workers=num_workers)


        # Training loop
        print(f"Starting training: \n")
        if kfold:
            print(
                f"  k-fold: {fold+1}/{n_splits}\n"
                f"  Training samples on this fold: {len(train_subsampler)}\n"
                f"  Testing samples on this fold: {len(test_subsampler)}\n"
                f"  Number of epochs: {num_epochs}\n"
                f"  Mini batch size: {batch_size}\n"
                f"  Number of batches: {len(trainloader)}\n"
                f"  Learning rate: {learning_rate})\n"
            )
        else:
            print(
                f"  Training samples: {len(dataset)}\n"
                f"  Number of epochs: {num_epochs}\n"
                f"  Mini batch size: {batch_size}\n"
                f"  Number of batches: {len(trainloader)}\n"
                f"  Learning rate: {learning_rate})\n"
            )
        loss_gen = []; loss_disc_real = []; loss_disc_fake = []; loss_mse=[]; loss_vgg=[]; psnrs=[];ssims=[]

        fig_advloss, ax_advloss = plt.subplots(dpi=300)
        fig_other_losses, ax_other_losses = plt.subplots(dpi=300)
        fig, ax = plt.subplots(1,2, dpi=300)
        fig_metrics, axs_metrics = plt.subplots(dpi=300)
        ax_advloss.set_xlim(0, num_epochs)
        ax_other_losses.set_xlim(0, num_epochs)
        axs_metrics.set_xlim(0, num_epochs)

        for epoch in range(num_epochs):
            for idx, (low_res, high_res) in enumerate(trainloader):

                # Send images to device
                high_res = high_res.to(device)
                low_res = low_res.to(device)

                # Generate fake (high_res) image from low_res
                fake = gen(low_res)

                loss_real, loss_fake = train_discriminator(disc, opt_disc, fake, high_res, bce)
                adv_loss, vgg_loss, mse_loss = train_generator(disc, opt_gen, fake, high_res, vgg_loss_fun, mse, bce)

                # Calculate PSNR and SSIM
                psnr = 0; ssim = 0
                for (hr, sr) in zip(high_res, fake):
                    psnr += peak_signal_noise_ratio(hr.detach().cpu().numpy(), sr.detach().cpu().numpy())
                    ssim += structural_similarity(hr.detach().cpu().numpy(), sr.detach().cpu().numpy(), channel_axis=0)
                psnr /= high_res.shape[0]
                ssim /= high_res.shape[0]

            # Append current epoch loss to list of losses
            loss_disc_real.append(float(loss_real.detach().cpu()))
            loss_disc_fake.append(float(loss_fake.detach().cpu()))
            loss_gen.append(float(adv_loss.detach().cpu())*1e3)
            loss_mse.append(float(mse_loss.detach().cpu()))
            loss_vgg.append(float(vgg_loss.detach().cpu())/0.006)
            psnrs.append(psnr)
            ssims.append(ssim)

            # Print progress every epoch
            print(
                f"Epoch [{epoch+1}/{num_epochs} - "
                f"Loss D: {loss_real+loss_fake:.3f}, Loss G: {adv_loss+mse_loss+vgg_loss:.3f},"
                f" Tra. PSNR: {psnr:.3f}, Tra. SSIM: {ssim:.3f}]"
                )

            # Plot adversarial loss
            x = np.arange(0, epoch+1)
            ax_advloss.plot(x, loss_gen, label='Gen. loss', color='r')
            ax_advloss.plot(x, loss_disc_real, label='Disc. loss (real)', color='k')
            ax_advloss.plot(x, loss_disc_fake, label='Disc. loss (fake)', color='b')
            ax_advloss.set_title('Generator and Discriminator losses vs. epoch')
            ax_advloss.set(xlabel='epoch')
            ax_advloss.set(ylabel='adversarial loss')
            if epoch == 0:
                ax_advloss.legend(loc='upper right')
                ax_advloss.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax_advloss.grid(visible=True)
            fig_advloss.savefig(os.path.join(config['figures']['dir'],'loss_evol.png'))

            # Plot MSE and VGG loss
                # ax_other_losses.xaxis.set_major_locator(MaxNLocator(integer=True))
            if epoch == 0:
                ax2 = ax_other_losses.twinx()
            line_mse, = ax_other_losses.plot(x, loss_mse, label='MSE Gen. loss', color='r')
            line_vgg, = ax2.plot(x, loss_vgg, label='VGG Gen. loss', color='r', ls='--')
            ax_other_losses.set_title('MSE and VGG Generator losses vs. epoch')
            ax_other_losses.set(xlabel='epoch')
            ax_other_losses.set(ylabel='MSE loss')
            ax2.set_ylabel('VGG loss', rotation=270, labelpad=14)
            if epoch == 0:
                ax_other_losses.legend(handles=[line_mse, line_vgg], loc='upper right')
                ax_other_losses.xaxis.set_major_locator(MaxNLocator(integer=True))
            # fig_other_losses.suptitle('MSE and VGG Generator losses vs. epoch')
            # Print progress every epoch
            ax_other_losses.grid(visible=True)
            fig_other_losses.savefig(os.path.join(config['figures']['dir'],'loss_evol_mse_vgg.png'))

            # Plot PSNR and SSIM 
            if epoch == 0:
                ax2_m = axs_metrics.twinx()
            line_psnr, = axs_metrics.plot(x, psnrs, label='PSNR', color='C0')
            line_ssim, = ax2_m.plot(x, ssims, label='SSIM', color='C1')
            axs_metrics.set_title('PSNR and SSIM on training data vs. epoch')
            axs_metrics.set(xlabel='epoch')
            axs_metrics.set(ylabel='PSNR')
            ax2_m.set_ylabel('SSIM', rotation=270, labelpad=14)
            # print(line_psnr.get_label())
            if epoch == 0:
                axs_metrics.legend(handles=[line_psnr, line_ssim], loc='lower right')
                axs_metrics.xaxis.set_major_locator(MaxNLocator(integer=True))
                # axs_metrics.xaxis.set_major_locator(MaxNLocator(integer=True))
            # Print progress every epoch
            axs_metrics.grid(visible=True)
            fig_metrics.savefig(os.path.join(config['figures']['dir'],'metrics_tra.png'))

            # Evaluate model
            sr_sample = fake[0].detach().cpu().numpy()
            hr_sample = high_res[0].detach().cpu().numpy()
            fig.suptitle(f'PSNR: {peak_signal_noise_ratio(hr_sample, sr_sample):.2f} | SSIM: {structural_similarity(hr_sample, sr_sample, channel_axis=0):.2f}', y=0.9)
            ax[0].imshow((np.moveaxis(sr_sample, 0, 2)*255).astype(np.uint8))
            ax[0].axis('off')
            ax[0].set_title('Super Resolution Image')
            ax[1].imshow((np.moveaxis(hr_sample, 0, 2)*255).astype(np.uint8))
            ax[1].axis('off')
            ax[1].set_title('High Resolution Image')
            fig.savefig(os.path.join('figures', 'lsh_train'),  bbox_inches='tight')

        # Save model at the end of all epochs
        if config['models']['save']:
            save_checkpoint(gen, opt_gen, filename=os.path.join(config['models']['rootdir'], config['models']['gen']))
            save_checkpoint(disc, opt_disc, filename=os.path.join(config['models']['rootdir'], config['models']['disc']))

        # Evaluation of this fold
        if kfold:
            print("\nEvaluation of cross validation dataset:")
            gen.eval()
            with torch.no_grad():
                # Iterate over the batches of test data and generate super resolution images from downscale of HR test images
                for idx, (low_res, high_res) in enumerate(testloader):
                    # Generate super resolution image from low_res
                    super_res = gen(low_res.to(device))
                    # Transfer images to cpu and convert them to arrays
                    super_res = np.asarray(super_res.cpu())
                    high_res = np.asarray(high_res.cpu())

                    if idx == 0 and test_sr:
                        sr_test = super_res[idx]
                        hr_test = high_res[idx]
                        fig, ax = plt.subplots(1,2)
                        fig.suptitle(f'PSNR: {peak_signal_noise_ratio(hr_test, sr_test):.2f} | SSIM: {structural_similarity(hr_test, sr_test, channel_axis=0):.2f}')
                        ax[0].imshow(np.moveaxis(sr_test, 0, 2))
                        ax[0].axis('off')
                        ax[0].set_title('Super Resolution Image')
                        ax[1].imshow(np.moveaxis(hr_test, 0, 2))
                        ax[1].axis('off')
                        ax[1].set_title('High Resolution Image')
                        plt.savefig(os.path.join('figures', 'lsh_train'),  bbox_inches='tight')
                    # Iterate over all images in this batch 
                    for i, hr_im in enumerate(high_res):
                        # Calculate PSNR
                        psnr[fold, idx, i] = peak_signal_noise_ratio(hr_im, super_res[i])
                        # Calculate SSIM
                        ssim[fold, idx, i] = structural_similarity(hr_im, super_res[i], channel_axis=0)

            # Print averaged PSNR and SSIM
            psnr[fold, :, :][psnr[fold, :, :] == 0] = np.nan
            ssim[fold, :, :][ssim[fold, :, :] == 0] = np.nan
            print(f"Average PSNR of fold {fold+1}/{n_splits}: {np.nanmean(psnr[fold, :, :]):.3f}")
            print(f"Average SSIM of fold {fold+1}/{n_splits}: {np.nanmean(ssim[fold, :, :]):.3f}")
            print(f"----------------------------------------")
            gen.train()

    if kfold:
        print(f"Average PSNR of all folds: {np.nanmean(psnr):.2f}")
        print(f"Average SSIM of all folds: {np.nanmean(ssim):.2f}")
        np.save("psnr.npy", psnr)
        np.save("ssim.npy", ssim)


if __name__ == "__main__":
    train_srgan(learning_rate, num_epochs, batch_size, num_workers)
    print("Finished training.")
