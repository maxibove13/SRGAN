# Super Resolution GAN

We also implement a basic NN and a CNN for practice.

## Questions

Why the discriminator loss for high_res is this one:   

```
loss_real = criterion(pred_real, torch.ones_like(pred_real) - 0.1 * torch.rand_like(pred_real))
```

Is it ok to sum discriminator real and fake loss?

Why gen loss is divided in vgg and adversarial_loss? Why adversarial loss is divided by 1000?

What is pin_memory in DataLoader?

What are optim.Adam betas?