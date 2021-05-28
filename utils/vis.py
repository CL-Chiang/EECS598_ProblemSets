import numpy as np
import os
import random

import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torchvision

"""
Utilities to help with visualizing images and other data
"""

def plot_gan_training(losses, title, fname):
    plt.figure()
    n_itr = len(losses)
    xs = np.arange(n_itr)

    plt.plot(xs, losses, label='loss')
    plt.legend()
    plt.title(title)
    plt.xlabel('Training Iteration')
    plt.ylabel('Loss')
    savefig(fname)

def plot_supervised(pretrained_losses, random_losses, title, fname):
    plt.figure()
    xs = np.arange(len(pretrained_losses))
    plt.plot(xs, pretrained_losses, label='bigan')
    xs = np.arange(len(random_losses))
    plt.plot(xs, random_losses, label='random init')
    plt.legend()
    plt.title(title)
    savefig(fname)

def plot_vae_training_plot(train_losses, test_losses, title, fname):
    elbo_train, recon_train, kl_train = train_losses[:, 0], train_losses[:, 1], train_losses[:, 2]
    elbo_test, recon_test, kl_test = test_losses[:, 0], test_losses[:, 1], test_losses[:, 2]
    plt.figure()
    n_epochs = len(test_losses) - 1
    x_train = np.linspace(0, n_epochs, len(train_losses))
    x_test = np.arange(n_epochs + 1)

    plt.plot(x_train, elbo_train, label='-elbo_train')
    plt.plot(x_train, recon_train, label='recon_loss_train')
    plt.plot(x_train, kl_train, label='kl_loss_train')
    plt.plot(x_test, elbo_test, label='-elbo_test')
    plt.plot(x_test, recon_test, label='recon_loss_test')
    plt.plot(x_test, kl_test, label='kl_loss_test')

    plt.legend()
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    savefig(fname)

def tensor_to_image(tensor):
    """
    Convert a torch tensor into a numpy ndarray for visualization.

    Inputs:
    - tensor: A torch tensor of shape (3, H, W) with elements in the range [0, 1]

    Returns:
    - ndarr: A uint8 numpy array of shape (H, W, 3)
    """
    tensor = tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0)
    ndarr = tensor.to("cpu", torch.uint8).numpy()
    return ndarr

def savefig(fname, show_figure=True):
    if not os.path.exists(os.path.dirname(fname)):
        os.makedirs(os.path.dirname(fname))
    plt.tight_layout()
    plt.savefig(fname)
    if show_figure:
        plt.show()

def show_samples(samples, fname=None, nrow=10, title='Samples', gray=False, hwc=False, figsize=None):
    samples = (torch.FloatTensor(samples))
    if hwc: 
      samples = samples.permute(0, 3, 1, 2)
    if gray: 
        samples = samples.unsqueeze(1).repeat(1, 3, 1, 1)
    grid_img = torchvision.utils.make_grid(samples, nrow=nrow)
    if figsize: 
        plt.figure(figsize=figsize)
    plt.title(title)
    plt.imshow((grid_img/255.0).permute(1, 2, 0))
    plt.axis('off')
    
    if fname is not None:
        savefig(fname)
    else:
        plt.show()

def save_training_plot(train_losses, test_losses, title, fname):
    plt.figure()
    n_epochs = len(test_losses) - 1
    x_train = np.linspace(0, n_epochs, len(train_losses))
    x_test = np.arange(n_epochs + 1)

    plt.plot(x_train, train_losses, label='train loss')
    plt.plot(x_test, test_losses, label='test loss')
    plt.legend()
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('NLL')
    savefig(fname)


