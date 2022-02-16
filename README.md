# Super Resolution GAN

A naive implementation of a Super-Resolution GAN.
Based on the work by Ledig et. al. 2017
arxiv.org/abs/1609.04802

## Questions

- Why the discriminator loss for high_res is this one:   

```
loss_real = criterion(pred_real, torch.ones_like(pred_real) - 0.1 * torch.rand_like(pred_real))
```

- Is it ok to sum discriminator real and fake loss?

- Why gen loss is divided in vgg and adversarial_loss? Why adversarial loss is divided by 1000?

- What is pin_memory in DataLoader?

- What are optim.Adam betas?