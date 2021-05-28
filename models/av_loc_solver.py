from tqdm import tqdm

import torchaudio

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

from utils.vis import show_samples

import numpy as np
import PIL
from PIL import Image
import pylab

from .av_loc_model import avolFeat

stft = torchaudio.transforms.MelSpectrogram(sample_rate=16000, hop_length=161, n_mels=64).cuda()

def log10(x): return torch.log(x)/torch.log(torch.tensor(10.))

def norm_range(x, min_val, max_val):
    return 2.*(x - min_val)/float(max_val - min_val) - 1.

def normalize_spec(spec, spec_min, spec_max):
    return norm_range(spec, spec_min, spec_max)

def db_from_amp(x, cuda=False):
    # rescale the audio
    if cuda: 
        return 20. * log10(torch.max(torch.tensor(1e-5).to('cuda'), x.float()))
    else: 
        return 20. * log10(torch.max(torch.tensor(1e-5), x.float()))

def audio_stft(stft, audio):
    # We'll apply stft to the audio samples to convert it to a HxW matrix
    N, C, A = audio.size()
    audio = audio.view(N * C, A)
    spec = stft(audio)
    spec = spec.transpose(-1, -2)
    spec = db_from_amp(spec, cuda=True)
    spec = normalize_spec(spec, -100., 100.)
    _, T, F = spec.size()
    spec = spec.view(N, C, T, F)
    return spec

def train(model, train_loader, optimizer, epoch, ttl_epochs=0, grad_clip=None, quiet=False):
    model.train()

    if not quiet:
        pbar = tqdm(total=len(train_loader.dataset))
    train_losses = []
    resol = 16

    if epoch < ttl_epochs // 2: 
        max_mode = False
    else: 
        max_mode = True

    for batch in train_loader:
        frame, audio = batch['frames'].cuda(), batch['audio'].cuda()
        audio = audio.unsqueeze(1)    
        spec = audio_stft(stft, audio)
        B = frame.shape[0]
        feat_img, feat_aud = model(frame, spec)

        feat_img = feat_img.reshape(B, 128, resol*resol).permute(0, 2, 1)
        feat_aud = feat_aud.squeeze().permute(1, 0)  
        
        """
        ------------- TODO ------------
        # Implement losses for training the audio-visual model
        # hint: Use torch.nn.functional.nll_loss
        """
        # feat_img.shape = N, p_size * p_size(256), 128
        #feat_aud.shape = 128, N
        
        if max_mode:
            nominator = torch.exp(torch.max(torch.einsum('nkc,cn->nk', feat_img, feat_aud) / 0.1, dim = 1)[0])
            #nominator = torch.max(nominator, dim = 1)[0]
            denominator = torch.sum(torch.exp(torch.max(torch.einsum('nkc,cb->nkb', feat_img, feat_aud) / 0.1, dim = 1)[0]), dim = 1)
            #denominator = torch.sum(torch.max(denominator, dim = 1)[0], dim = 1)
            loss = -torch.mean(torch.log((nominator + 1e-6) / (denominator + 1e-6)))
        else:
            nominator = torch.einsum('nkc,cn->nk', feat_img, feat_aud) / 256
            nominator = torch.sum(nominator, dim = 1)
            
            denominator = torch.einsum('nkc,cb->nb', feat_img, feat_aud) / 256
            
            nominator = torch.exp(nominator/0.1)
            denominator = torch.exp(denominator/0.1)
            denominator = torch.sum(denominator, dim = 1) 
            loss = -torch.mean(torch.log((nominator + 1e-6) / (denominator + 1e-6)))


        optimizer.zero_grad()

        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        train_losses.append(loss.item())

        if not quiet:
            pbar.set_description(f'Epoch {epoch} loss: {loss:.4f}')
            pbar.update(frame.shape[0])
    if not quiet:
        pbar.close()
    return train_losses


def eval_loss(model, data_loader, ttl_epochs=0, quiet=False, epoch=0):
    model.eval()
    total_losses = 0
    resol = 16

    if epoch < ttl_epochs // 2: 
        max_mode = False
    else: 
        max_mode = True

    with torch.no_grad():
        for batch in data_loader:
            frame, audio = batch['frames'].cuda(), batch['audio'].cuda()
            audio = audio.unsqueeze(1)    
            spec = audio_stft(stft, audio)
            B = frame.shape[0]
            feat_img, feat_aud = model(frame, spec)

            feat_img = feat_img.reshape(B, 128, resol*resol).permute(0, 2, 1)
            feat_aud = feat_aud.squeeze().permute(1, 0)
            """
            ------------- TODO ------------
            # Implement losses of the audio-visual model
            # hint: Use torch.nn.functional.nll_loss
            """
            if max_mode:
                nominator = torch.max(torch.einsum('nkc,cn->nk', feat_img, feat_aud) / 0.1, dim = 1)[0]
                #nominator = torch.max(nominator, dim = 1)[0]
                denominator = torch.sum(torch.max(torch.einsum('nkc,cb->nkb', feat_img, feat_aud) / 0.1, dim = 1)[0], dim = 1)
                #denominator = torch.sum(torch.max(denominator, dim = 1)[0], dim = 1)
                loss = -torch.mean(torch.log(nominator / denominator))
            else:
                nominator = torch.einsum('nkc,cn->nk', feat_img, feat_aud) / 256
                nominator = torch.sum(nominator, dim = 1)
                
                denominator = torch.einsum('nkc,cb->nb', feat_img, feat_aud) / 256
            
                nominator = torch.exp(nominator/0.1)
                denominator = torch.exp(denominator/0.1)
                denominator = torch.sum(denominator, dim = 1) 
                loss = -torch.mean(torch.log(nominator / denominator))
            

            total_losses += loss 
            
        avg_loss = total_losses / len(data_loader)
        if not quiet: 
            print(f'{avg_loss:.4f}')

    return avg_loss


def train_epochs(model, train_loader, test_loader, train_args, exp_path=None, quiet=False):
    epochs, lr = train_args['epochs'], train_args['lr']
    grad_clip = train_args.get('grad_clip', None)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    train_losses = []
    test_losses = []
    for epoch in range(epochs):
        model.train()
        train_losses.extend(train(model, train_loader, optimizer, epoch, ttl_epochs=epochs, grad_clip=None, quiet=quiet))
        test_loss = eval_loss(model, test_loader, ttl_epochs=epochs, epoch=epoch, quiet=quiet)
        test_losses.append(test_loss)
        if not quiet:
            print(f'Epoch {epoch}, Test loss {test_loss:.4f}')
        ssl_results = loc_sound_source(test_loader, model)
        ssl_results = np.array(ssl_results, dtype='uint8').transpose(0, 3, 1, 2)
        ssl_imgs = ssl_results[:100]
        show_samples(ssl_imgs, fname=f'{exp_path}/av_loc_samples_ep{epoch}.png', 
                     title='sound source localization', nrow=10, figsize=(7,7))

    return train_losses, test_losses


def clip_rescale_torch(x, lo = None, hi = None):
    # rescale torch tensor
    if lo is None:
        lo = torch.min(x)
    if hi is None:
        hi = torch.max(x)
    return torch.clamp((x - lo)/(hi - lo), 0., 1.)


def clip_rescale(x, lo = None, hi = None):
    # rescale numpy array
    if lo is None:
        lo = np.min(x)
    if hi is None:
        hi = np.max(x)
    return np.clip((x - lo)/(hi - lo), 0., 1.)

def apply_cmap(im, cmap = pylab.cm.jet, lo = None, hi = None):
    return cmap(clip_rescale(im, lo, hi).flatten()).reshape(im.shape[:2] + (-1,))[:, :, :3]

def cmap_im(cmap, im, lo = None, hi = None):
    return np.uint8(255*apply_cmap(im, cmap, lo, hi))

def load_img(frame_path): 
    image = Image.open(frame_path).convert('RGB')
    image = image.resize((256, 256), resample=PIL.Image.BILINEAR)
    image = np.array(image)

    img_h = image.shape[0]
    img_w = image.shape[1]

    return image, img_h, img_w

def loc_sound_source(data_loader, model): 
    """
    Visualize the sound source. 
    Input: 
    - data_loader: data loader for audio-visual samples 
    - model: avolFeat model 
    outpuy: 
    - out_imgs: a numpy array of shape (N, C, H, W), which saves sound source localization results
    """
    resol = 16
    with torch.no_grad():
        out_imgs = []
        for batch in data_loader:
            frame, audio = batch['frames'].cuda(), batch['audio'].cuda()
            frame_paths = batch['frame_info']
            audio = audio.unsqueeze(1)    
            spec = audio_stft(stft, audio)
            B = frame.shape[0]
            feat_img, feat_aud = model(frame, spec)
            feat_img = feat_img.reshape(B, 128, resol*resol).permute(0, 2, 1)

            # compute similarity scores between different image regions and audio
            sim = torch.bmm(feat_img, feat_aud).squeeze()
            sim = sim.view(B, 1, resol, resol)
            # upsample the similarity score map
            sim_up = nn.functional.interpolate(sim, size=((256, 256)), mode='bilinear', align_corners=False).squeeze()

            for ind in range(B):
                im, img_h, img_w = load_img(frame_paths[ind])

                pred_prob = sim_up[ind]
                pred_prob = clip_rescale_torch(pred_prob)
                pred_prob = pred_prob.cpu().numpy()

                # convert similarity score according to jet colormap
                vis = cmap_im(pylab.cm.jet, pred_prob, lo=0, hi=1)
                p = pred_prob / pred_prob.max() * 0.5 + 0.1
                p = p[..., None]

                outs = np.uint8(im*(1-p) + vis*p)
                out_imgs.append(outs)
            break
    return out_imgs

def train_av_loc(train_dataset, test_dataset, exp_path=None, quiet=False): 
    """
    A function completing the model training, testing, sampling.  
    Input: 
    - train_data: an ImgAudDset object for training
    - test_data: an ImgAudDset object for testing
    - exp_path: path to save the visualization results
    - quiet: whether to show the training process
    return: 
    - train_losses: a list recording the losses during training
    - test_losses: a list recording the losses during testing
    - ssl_results: a numpy array of shape (N, H, W, C), which saves sound source localization results
    """
    model = avolFeat().cuda()
    train_loader = data.DataLoader(train_dataset, batch_size=128, shuffle=True, 
                                   drop_last=True, num_workers=2, pin_memory=True)
    test_loader = data.DataLoader(test_dataset, batch_size=128, num_workers=2, 
                                  pin_memory=True)

    train_losses, test_losses = train_epochs(model, train_loader, test_loader,
                                             dict(epochs=50, lr=1e-4), exp_path=exp_path, quiet=quiet)
    ssl_results = loc_sound_source(test_loader, model)
    ssl_results = np.array(ssl_results, dtype='uint8').transpose(0, 3, 1, 2)

    return train_losses, test_losses, ssl_results