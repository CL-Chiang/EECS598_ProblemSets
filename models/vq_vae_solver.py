from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.utils.data as data
import torch.optim as optim
import torch.nn.functional as F

from .vq_vae_model import VQVAENet, PixelCNN, vq_vae_loss


def train(model, train_loader, optimizer, epoch, quiet, grad_clip=None):
    model.train()

    if not quiet:
        pbar = tqdm(total=len(train_loader.dataset))
    losses = OrderedDict()
    for x in train_loader:
        x = x.cuda()
        x_tilde, diff = model(x)
        loss_dict = vq_vae_loss(x_tilde, diff, x)
        optimizer.zero_grad()

        loss_dict['loss'].backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        desc = f'Epoch {epoch}'
        for k, v in loss_dict.items():
            if k not in losses:
                losses[k] = []
            losses[k].append(v.item())
            avg_loss = np.mean(losses[k][-50:])
            desc += f', {k} {avg_loss:.4f}'

        if not quiet:
            pbar.set_description(desc)
            pbar.update(x.shape[0])
    if not quiet:
        pbar.close()
    return losses


def eval_loss(model, data_loader, quiet):
    model.eval()
    total_losses = OrderedDict()
    with torch.no_grad():
        for x in data_loader:
            x = x.cuda()
            x_tilde, diff = model(x)
            loss_dict = vq_vae_loss(x_tilde, diff, x)
            for k, v in loss_dict.items():
                total_losses[k] = total_losses.get(k, 0) + v.item() * x.shape[0]

        desc = 'Test '
        for k in total_losses.keys():
            total_losses[k] /= len(data_loader.dataset)
            desc += f', {k} {total_losses[k]:.4f}'
        if not quiet:
            print(desc)
    return total_losses

def train_epochs(model, train_loader, test_loader, train_args, quiet=False):
    """
    Implementation for training VQ-VAE model. 

    Inputs: 
    - model: VQ-VAE model 
    - train_loader: dataloader for training set
    - test_loader: dataloader for testing set
    - train_args: an OrderedDict saving training parameters
    """
    epochs, lr = train_args['epochs'], train_args['lr']
    grad_clip = train_args.get('grad_clip', None)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Implement the training and testing of model. 
    # Record the losses during training and testing
    train_losses, test_losses = OrderedDict(), OrderedDict()
    for epoch in range(epochs):
        model.train()
        train_loss = train(model, train_loader, optimizer, epoch, quiet, grad_clip)
        test_loss = eval_loss(model, test_loader, quiet)

        for k in train_loss.keys():
            if k not in train_losses:
                train_losses[k] = []
                test_losses[k] = []
            train_losses[k].extend(train_loss[k])
            test_losses[k].append(test_loss[k])

    return train_losses, test_losses


def train_pixcnn(model, train_loader, optimizer, epoch, quiet, grad_clip=None):
    model.train()

    if not quiet:
        pbar = tqdm(total=len(train_loader.dataset))
    losses = OrderedDict()
    for x in train_loader:
        x = x.cuda()
        loss_dict = model.loss(x)
         
        optimizer.zero_grad()

        loss_dict['loss'].backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        desc = f'Epoch {epoch}'
        for k, v in loss_dict.items():
            if k not in losses:
                losses[k] = []
            losses[k].append(v.item())
            avg_loss = np.mean(losses[k][-50:])
            desc += f', {k} {avg_loss:.4f}'

        if not quiet:
            pbar.set_description(desc)
            pbar.update(x.shape[0])
    if not quiet:
        pbar.close()
    return losses


def eval_loss_pixcnn(model, data_loader, quiet):
    model.eval()
    total_losses = OrderedDict()
    with torch.no_grad():
        for x in data_loader:
            x = x.cuda()
            loss_dict = model.loss(x)
            for k, v in loss_dict.items():
                total_losses[k] = total_losses.get(k, 0) + v.item() * x.shape[0]

        desc = 'Test '
        for k in total_losses.keys():
            total_losses[k] /= len(data_loader.dataset)
            desc += f', {k} {total_losses[k]:.4f}'
        if not quiet:
            print(desc)
    return total_losses


def train_epochs_pixcnn(model, train_loader, test_loader, train_args, quiet=False):
    """
    Implementation for training PixelCNN model. 

    Inputs: 
    - model: PixelCNN model 
    - train_loader: dataloader for training set
    - test_loader: dataloader for testing set
    - train_args: an OrderedDict saving training parameters
    """
    epochs, lr = train_args['epochs'], train_args['lr']
    grad_clip = train_args.get('grad_clip', None)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Implement the training and testing of model. 
    # Record the losses during training and testing
    train_losses, test_losses = OrderedDict(), OrderedDict()
    for epoch in range(epochs):
        model.train()
        train_loss = train_pixcnn(model, train_loader, optimizer, epoch, quiet, grad_clip)
        test_loss = eval_loss_pixcnn(model, test_loader, quiet)

        for k in train_loss.keys():
            if k not in train_losses:
                train_losses[k] = []
                test_losses[k] = []
            train_losses[k].extend(train_loss[k])
            test_losses[k].append(test_loss[k])
    return train_losses, test_losses


def train_vq_vae(train_data, test_data): 
    """
    A function completing the VQ-VAE model training, testing, sampling.  
    Input: 
    - train_data: a numpy array of image for training
    - test_data: a numpy array of image for testing
    return: 
    - vqvae_train_losses: a Python list containing losses during the training of VQ-VAE (per mini-batch) 
    - vqvae_test_losses: a Python list containing losses during testing of VQ-VAE (per epoch)
    - prior_train_losses: a Python list containing losses during the training of PixelCNN (per mini-batch) 
    - prior_test_losses: a Python list containing losses during testing of PixelCNN (per epoch)
    - samples: numpy array of shape (N, H, W, C), containing random samples generated from the trained model. 
    - reconstructions: a numpy array of shape (N, C, H, W), which saves original images and reconstructed image pairs 
    
    ------- Instructions -------
    
    To train a model, we'll 
    - Construct a dataset using *torch.utils.data.DataLoader*. 
      Normalize the input to [-1, 1]
      The output of the Dataloader is images without labels. 
    - Construct a VQ-VAE model and loss (located in *models/vq_vae_model.py*) following the instruction in the file. 
      and remember to use GPU as the accelerator
    - Train the VQ-VAE model. When training, it's recommended to clip the gradient. 
      Consider using *torch.nn.utils.clip_grad_norm_* and set the max_norm to 1. 
    - Construct a prior dataset for the PixelCNN model. 
      Specifically, use the encoder of VQ-VAE model to extract *the indices* of discrete latent variables for each sample. 
    - Construct a PixelCNN model (located in *models/vq_vae_model.py*)
    - Construct training and testing set for prior dataset using *torch.utils.data.DataLoader*. 
    - Train the PixelCNN model. When training, it's recommended to clip gradient. 
      Consider using *torch.nn.utils.clip_grad_norm_* and set the max_norm to 1. 
    - Get the samples from the trained model. 
      To do so, you can first use PixelCNN model to sample priors, 
      then use the decoder of VQ-VAE model to generate the images
    - Get original and reconstructed image pairs. 

    Recommended hyperparameters: 
    - For VQ-VAE model: 
        - Batch size: 128
        - Learning rate: 0.001
        - Total epochs: 20
        - Adam optimizer 
    - For PixelCNN model: 
        - Batch size: 128
        - Learning rate: 0.001
        - Total epochs: 15
        - Adam optimizer 
    """
    model = VQVAENet(256, 128).cuda()
    train_loader = data.DataLoader(train_data, batch_size=128, shuffle=True)
    test_loader = data.DataLoader(test_data, batch_size=128)
    train_losses, test_losses = train_epochs(model, train_loader, test_loader,
                                              dict(epochs=20, lr=1e-3, grad_clip=1), quiet=False)
    vqvae_train_losses, vqvae_test_losses = train_losses['loss'], test_losses['loss']

    def create_prior_dataset(data_loader):
        prior_data = []
        with torch.no_grad():
            for x in data_loader:
                x = x.cuda()
                z = model.encode_code(x)
                prior_data.append(z.long())
        return torch.cat(prior_data, dim=0)

    prior = PixelCNN(code_size=128, input_shape=(8, 8), dim=128, n_layers=10).cuda()
    prior_train_data, prior_test_data = create_prior_dataset(train_loader), create_prior_dataset(test_loader)
    prior_train_loader = data.DataLoader(prior_train_data, batch_size=128, shuffle=True)
    prior_test_loader = data.DataLoader(prior_test_data, batch_size=128)

    prior_train_losses, prior_test_losses = train_epochs_pixcnn(prior, prior_train_loader, prior_test_loader,
                                                          dict(epochs=15, lr=1e-3, grad_clip=1), quiet=False)
    prior_train_losses, prior_test_losses = prior_train_losses['loss'], prior_test_losses['loss']

    samples = prior.sample(100).long()
    samples = model.decode_code(samples) * 255

    x = next(iter(test_loader))[:50].cuda()
    with torch.no_grad():
        z = model.encode_code(x)
        x_recon = model.decode_code(z)
    x = x.cpu().permute(0, 2, 3, 1).numpy()
    reconstructions = np.stack((x, x_recon), axis=1).reshape((-1, 32, 32, 3)) * 255
    
    return vqvae_train_losses, vqvae_test_losses, prior_train_losses, prior_test_losses, samples, reconstructions


