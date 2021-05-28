from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F

from .pixel_cnn_model import PixelCNN, pixel_cnn_loss


def train(model, train_loader, optimizer, epoch, grad_clip=None, quiet=False):
    model.train()
    if not quiet:
        pbar = tqdm(total=len(train_loader.dataset))
    train_losses = []
    for x in train_loader:
        x = x.cuda().contiguous()
        out = model(x)
        loss = pixel_cnn_loss(out, x)
        optimizer.zero_grad()
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        train_losses.append(loss.item())
        if not quiet:
            pbar.set_description(f'Epoch {epoch} loss: {loss:.4f}')
            pbar.update(x.shape[0])
    if not quiet:
        pbar.close()
    return train_losses

def eval_loss(model, data_loader, quiet=False):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x in data_loader:
            x = x.cuda().contiguous()
            out = model(x)
            loss = pixel_cnn_loss(out, x)
            total_loss += loss * x.shape[0]
        avg_loss = total_loss / len(data_loader.dataset)
        if not quiet: 
            print(f'{avg_loss:.4f}')
    return avg_loss.item()

def train_epochs(model, train_loader, test_loader, train_args, quiet=False):
    # implement the training of model. 
    # Record the negative log-likelihood during training and testing

    epochs, lr = train_args['epochs'], train_args['lr']
    grad_clip = train_args.get('grad_clip', None)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    test_losses = [eval_loss(model, test_loader, quiet=quiet)]
    for epoch in range(epochs):
        model.train()
        train_losses.extend(train(model, train_loader, optimizer, epoch, grad_clip, quiet=quiet))
        test_loss = eval_loss(model, test_loader, quiet=quiet)
        test_losses.append(test_loss)
        if not quiet:
            print(f'Epoch {epoch}, Test loss {test_loss:.4f}')

    return train_losses, test_losses


def train_pixel_cnn(train_data, test_data, quiet=False): 
    """
    A function completing the model training, testing, sampling.  
    Input: 
    - train_data: a numpy array of image for training
    - test_data: a numpy array of image for testing
    return: 
    - nll_train: a list recording the negative log-likelihood during training
    - nll_test: a list recording the negative log-likelihood during testing
    - samples: a numpy array of shape (N, H, W, C), which saves samples generated by the trained model
    
    ------- Instructions -------
    To train a model, we'll 
    - Construct a dataset using *torch.utils.data.DataLoader*. 
      The output of the Dataloader is images without labels. 
      Normalize the input to [-1, 1]
    - Construct a PixelCNN model and loss (located in *models/pixel_cnn_model.py*) following the instruction in the file. 
    - Train the PixelCNN model. 
    - Samples from the PixelCNN model. 
    
    Recommended hyperparameters: 
    - Batch size: 128
    - Learning rate: 0.001
    - Total epochs: 10
    - Adam optimizer 
    """
    H, W = 28, 28
    model = PixelCNN((1, H, W), 2, n_layers=5).cuda()
    train_loader = data.DataLoader(np.uint8(train_data), batch_size=128, shuffle=True)
    test_loader = data.DataLoader(np.uint8(test_data), batch_size=128)

    nll_train, nll_test = train_epochs(model, train_loader, test_loader,
                                             dict(epochs=10, lr=1e-3), quiet=quiet)
    samples = model.sample(100)

    return nll_train, nll_test, samples





