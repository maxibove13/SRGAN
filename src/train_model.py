#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Script to train a SRGAN"""

__author__ = "Maximiliano Bove"
__email__ = "maxibove13@gmail.com"
__status__ = "Development"
__date__ = "02/22"


# Built-in modules
import os
import time

# Third party modules
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import cm
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
from models.model import Generator, Discriminator, VGGLoss
from utils import load_checkpoint, save_checkpoint, TestingSet

# read yaml file
with open('config.yaml') as file:
    config = yaml.safe_load(file)
# By default hyperparameters are defined by config.yaml
learning_rate = config['train']['learning_rate']
num_epochs = config['train']['num_epochs']
batch_size = config['train']['batch_size']
num_workers = config['train']['num_workers']
multibatch = config['train']['multibatch']

def train_srgan(learning_rate, num_epochs, batch_size, num_workers):

    # Print test comparison figure between SR and HR.
    test_sr = True

    # Define device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(
        f"Using device: {torch.cuda.get_device_name()}"
        f" with {torch.cuda.get_device_properties(0).total_memory/1024/1024/1024:.0f} GiB"
        )

    # Import training and testing dataset
    print("Importing dataset...")
    dataset = ImageDataset(root_dir=os.path.join(config['data']['rootdir'], config['data']['dataset'], 'train', 'dataset'))
    print(f"{len(dataset)} training samples from {dataset.root_dir}")
    testing_dataset = ImageDataset(root_dir=os.path.join(config['data']['rootdir'], config['data']['dataset'], 'test', 'dataset'))
    print(f"{len(testing_dataset)} testing samples from {dataset.root_dir}")

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
        # load_checkpoint(
        #     os.path.join(config['models']['rootdir'], config['models']['disc']), disc, opt_disc, learning_rate, 
        #     device
        # )

    # initialize kfold validation
    kfold = config['validation']['kfold']
    if kfold:
        n_splits = config['validation']['n_splits']
        kfold = KFold(n_splits=n_splits, shuffle=True)
        psnr = np.zeros((n_splits, len(dataset)//n_splits//batch_size + 1, batch_size))
        ssim = np.zeros((n_splits, len(dataset)//n_splits//batch_size + 1, batch_size))

    # Iterate over kfolds only if kfold is on.
    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)) if kfold else enumerate([(0,0)]):

        if kfold:
            train_subsampler = SubsetRandomSampler(train_idx)
            val_subsampler = SubsetRandomSampler(val_idx)
            # Load data
            print(f"\nSpliting and loading datasets [{fold+1}/{n_splits}]...")
            trainloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True, num_workers=num_workers, sampler=train_subsampler)
            valloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True, num_workers=num_workers, sampler=val_subsampler)
        else:
            print(f"\nLoading dataloaders...")
            trainloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True, num_workers=num_workers)
            testloader = DataLoader(testing_dataset, batch_size=batch_size, pin_memory=True, num_workers=num_workers)


        # Training loop
        print(f"Starting training: \n")
        if kfold:
            print(
                f"  k-fold: {fold+1}/{n_splits}\n"
                f"  Training samples on this fold: {len(train_subsampler)}\n"
                f"  Testing samples on this fold: {len(val_subsampler)}\n"
                f"  Number of epochs: {num_epochs}\n"
                f"  Mini batch size: {batch_size}\n"
                f"  Number of batches: {len(trainloader)}\n"
                f"  Learning rate: {learning_rate})\n"
            )
        else:
            print(
                f"  Training samples: {len(dataset)}\n"
                f"  Testing samples : {len(testing_dataset)}\n"
                f"  Number of epochs: {num_epochs}\n"
                f"  Mini batch size: {batch_size}\n"
                f"  Number of batches: {len(trainloader)}\n"
                f"  Learning rate: {learning_rate})\n"
            )
        loss_gen = []; loss_disc_real = []; loss_disc_fake = []; loss_mse=[]; loss_vgg=[]; psnrs=[]; ssims=[]; psnrs_test=[];ssims_test=[]

        fig_advloss, ax_advloss = plt.subplots(dpi=300)
        fig_other_losses, ax_other_losses = plt.subplots(dpi=300)
        fig, ax = plt.subplots(1,2, dpi=300)
        fig_metrics, axs_metrics = plt.subplots(dpi=300)
        ax_advloss.set_xlim(0, num_epochs)
        ax_other_losses.set_xlim(0, num_epochs)
        axs_metrics.set_xlim(0, num_epochs)

        # TRAINING LOOP
        for epoch in range(num_epochs):
            gen.train()
            # for each mini batch if multibatch
            psnr_b, ssim_b = 0, 0
            for idx, (low_res, high_res) in enumerate(trainloader) if multibatch else enumerate([(0,0)]):

                if not multibatch:
                    low_res, high_res = next(iter(trainloader))

                # Send images to device
                high_res = high_res.to(device)
                low_res = low_res.to(device)

                # Generate high_res image from low_res (fake image)
                fake = gen(low_res)

                # TRAIN DISCRIMINATOR
                # Train on real data
                pred_real = disc(high_res)
                loss_real = bce(pred_real, torch.ones_like(pred_real) - 0.1 * torch.rand_like(pred_real))
                # Train on fake data
                pred_fake = disc(fake.detach())
                loss_fake = bce(pred_fake, torch.zeros_like(pred_fake))
                loss_disc = (loss_real + loss_fake)
                # Backward pass
                opt_disc.zero_grad()
                loss_disc.backward()
                # Update weights
                opt_disc.step()

                # TRAIN GENERATOR
                pred_fake = disc(fake)
                adv_loss = 1e-3 * bce(pred_fake, torch.ones_like(pred_fake))
                vgg_loss = 0.006 * vgg_loss_fun(fake, high_res)
                mse_loss = mse(fake, high_res)
                loss = adv_loss + vgg_loss + mse_loss
                opt_gen.zero_grad()
                # Backward pass
                loss.backward()
                # Update weights
                opt_gen.step()

                # Calculate PSNR and SSIM
                psnr = 0; ssim = 0
                for (hr, sr) in zip(high_res, fake):
                    psnr += peak_signal_noise_ratio(hr.detach().cpu().numpy()[0], sr.detach().cpu().numpy()[0])
                    ssim += structural_similarity(hr.detach().cpu().numpy()[0], sr.detach().cpu().numpy()[0])
                psnr /= len(high_res); psnr_b += psnr
                ssim /= len(high_res); ssim_b += ssim

            if multibatch:
                psnr_b /= len(trainloader)
                ssim_b /= len(trainloader)


            # evaluation of the model on the testing data. This is for tuning hyperparameter to assess the regularization of the model
            gen.eval()
            psnr_test_b, ssim_test_b = 0, 0
            # for each mini batch in testloader if multibatch
            for idx, (low_res_test, high_res_test) in enumerate(testloader) if multibatch else enumerate([(0,0)]):

                if not multibatch:
                    low_res_test, high_res_test = next(iter(testloader))

                with torch.no_grad():
                    synths = gen(low_res_test.to(device))
                    # Iterate over all images in this batch
                    psnr_test, ssim_test = 0, 0 
                    for i, (synth, hr_test) in enumerate(zip(synths,high_res_test)):
                        psnr_test += peak_signal_noise_ratio(hr_test.numpy()[0], synth.cpu().numpy()[0])
                        ssim_test += structural_similarity(hr_test.numpy()[0], synth.cpu().numpy()[0])
                    psnr_test /= len(synths); psnr_test_b += psnr_test
                    ssim_test /= len(synths); ssim_test_b += ssim_test
            
            if multibatch:
                psnr_test_b /= len(testloader)
                ssim_test_b /= len(testloader)


            # Append current epoch loss to list of losses
            loss_disc_real.append(float(loss_real.detach().cpu()))
            loss_disc_fake.append(float(loss_fake.detach().cpu()))
            loss_gen.append(float(adv_loss.detach().cpu())*1e3)
            loss_mse.append(float(mse_loss.detach().cpu()))
            loss_vgg.append(float(vgg_loss.detach().cpu())/0.006)
            psnrs.append(psnr_b)
            ssims.append(ssim_b)
            psnrs_test.append(psnr_test_b)
            ssims_test.append(ssim_test_b)

            # Print progress every epoch
            print(
                f"[{epoch+1:03d}/{num_epochs} - "
                f"Loss: D: {loss_real+loss_fake:.3f}, G: {adv_loss+mse_loss+vgg_loss:.3f},"
                f" Tra.: PSNR: {psnr_b:.3f}, SSIM: {ssim_b:.3f}],"
                f" Test.: PSNR: {psnr_test_b:.3f}, SSIM: {ssim_test_b:.3f}], "
                f"{torch.cuda.memory_allocated(device=device)/1024/1024/1024:.1f}GiB alloc."
                )

            # Plot adversarial loss
            x = np.arange(0, epoch+1)
            ax_advloss.plot(x, loss_gen, label='Generator loss', color='r')
            ax_advloss.plot(x, loss_disc_real, label='Discriminator loss (real)', color='k')
            ax_advloss.plot(x, loss_disc_fake, label='Discriminator loss (synthetic)', color='b')
            # ax_advloss.set_title('Generator and Discriminator losses vs. epoch')
            ax_advloss.set(xlabel='epoch')
            ax_advloss.set(ylabel='adversarial loss')
            ax_advloss.set_ylim(top=15)
            # ax_advloss.set_ylim(bottom=0, top=10)
            if epoch == 0:
                ax_advloss.legend(loc='upper right')
                ax_advloss.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax_advloss.grid(visible=True)
            fig_advloss.savefig(os.path.join(config['figures']['dir'],'loss_evol.png'), dpi=300, bbox_inches='tight')

            # Plot MSE and VGG loss
                # ax_other_losses.xaxis.set_major_locator(MaxNLocator(integer=True))
            if epoch == 0:
                ax2 = ax_other_losses.twinx()
            line_mse, = ax_other_losses.plot(x, loss_mse, label='MSE loss', color='r')
            line_vgg, = ax2.plot(x, loss_vgg, label='VGG loss', color='b', ls='-')
            # ax_other_losses.set_title('MSE and VGG Generator losses vs. epoch')
            ax_other_losses.set(xlabel='epoch')
            ax_other_losses.set(ylabel='MSE loss')
            ax2.set_ylabel('VGG loss', rotation=270, labelpad=14)
            if epoch == 0:
                ax_other_losses.legend(handles=[line_mse, line_vgg], loc='upper right')
                ax_other_losses.xaxis.set_major_locator(MaxNLocator(integer=True))
            # fig_other_losses.suptitle('MSE and VGG Generator losses vs. epoch')
            # Print progress every epoch
            ax_other_losses.grid(visible=True)
            fig_other_losses.savefig(os.path.join(config['figures']['dir'],'loss_evol_mse_vgg.png'), dpi=300, bbox_inches='tight')

            # Plot PSNR and SSIM 
            if epoch == 0:
                ax2_m = axs_metrics.twinx()
            line_psnr, = axs_metrics.plot(x, psnrs, label='PSNR training set', color='C0')
            # line_psnr_test, = axs_metrics.plot(x, psnrs_test, label='PSNR testing set', color='C0', ls='-.')
            line_ssim, = ax2_m.plot(x, ssims, label='SSIM training set', color='C1')
            # line_ssim_test, = ax2_m.plot(x, ssims_test, label='SSIM testing set', color='C1', ls='-.')
            # axs_metrics.set_title('PSNR and SSIM on training data vs. epoch')
            axs_metrics.set(xlabel='epoch')
            axs_metrics.set(ylabel='PSNR')
            ax2_m.set_ylabel('SSIM', rotation=270, labelpad=14)
            if epoch == 0:
                axs_metrics.legend(handles=[line_psnr, line_ssim], loc='lower right')
                axs_metrics.xaxis.set_major_locator(MaxNLocator(integer=True))
                # axs_metrics.xaxis.set_major_locator(MaxNLocator(integer=True))
            # Print progress every epoch
            axs_metrics.grid(visible=True)
            fig_metrics.savefig(os.path.join(config['figures']['dir'],'psnr_ssim_per_epoch.png'), dpi=300, bbox_inches='tight')

            # Evaluate model on training images
            # sr_sample = fake[0].detach().cpu().numpy()
            # hr_sample = high_res[0].detach().cpu().numpy()
            # fig.suptitle(f'PSNR: {peak_signal_noise_ratio(hr_sample[0], sr_sample[0]):.2f} | SSIM: {structural_similarity(hr_sample[0], sr_sample[0]):.2f}', y=0.9)
            # ax[0].imshow((np.moveaxis(sr_sample, 0, 2))[:,:,0], cmap=cm.gray)
            # ax[0].axis('off')
            # ax[0].set_title('Super Resolution Image')
            # ax[1].imshow((np.moveaxis(hr_sample, 0, 2))[:,:,0], cmap=cm.gray)
            # ax[1].axis('off')
            # ax[1].set_title('High Resolution Image')
            # fig.savefig(os.path.join('figures', 'lsh_train'),  bbox_inches='tight')


            # evaluate model on testing set
            if epoch == 0:
                testing_set = TestingSet(
                    os.path.join(config['data']['rootdir'], config['data']['dataset'], 'test', 'dataset'),
                    device,
                )
            if epoch % 10 == 0:
                testing_set.plot_grid_of_samples(gen)
                testing_set.plot_grid_of_errors(gen)

        # Save model at the end of all epochs
        if config['models']['save']:
            save_checkpoint(gen, opt_gen, filename=os.path.join(config['models']['rootdir'], config['models']['gen']))
            save_checkpoint(disc, opt_disc, filename=os.path.join(config['models']['rootdir'], config['models']['disc']))

        # Evaluation of this fold (kfold is done to report final model performance)
        if kfold:
            print("\nEvaluation of cross validation dataset:")
            gen.eval()
            with torch.no_grad():
                # Iterate over the batches of test data and generate super resolution images from downscale of HR test images
                for idx, (low_res, high_res) in enumerate(valloader):
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
                        ax[0].imshow(np.moveaxis(sr_test, 0, 2)[:, :, 0])
                        ax[0].axis('off')
                        ax[0].set_title('Super Resolution Image')
                        ax[1].imshow(np.moveaxis(hr_test, 0, 2)[:, :, 0])
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
    tic = time.time()
    train_srgan(learning_rate, num_epochs, batch_size, num_workers)
    toc = time.time()
    print("Finished training.")
    print(f"Training duration: {((toc-tic)/60):.2f} m ")
