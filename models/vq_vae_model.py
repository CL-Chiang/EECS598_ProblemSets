from collections import OrderedDict
import numpy as np

import torch
import torch.nn as nn

import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 1)
        )
      
    def forward(self, x):
        return x + self.net(x)


class LayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        x_shape = x.shape
        x = super().forward(x)
        return x.permute(0, 3, 1, 2).contiguous()


class MaskConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        assert mask_type == 'A' or mask_type == 'B'
        super().__init__(*args, **kwargs)
        self.register_buffer('mask', torch.zeros_like(self.weight))
        self.create_mask(mask_type)

    def forward(self, input):
        out = F.conv2d(input, self.weight * self.mask, self.bias, self.stride,
                       self.padding, self.dilation, self.groups)
        return out

    def create_mask(self, mask_type):
        # ----------------- TODO ------------------ #
        # Implement Mask for type A and B layer here 
        # ----------------------------------------- #
        _, _, height, width = self.weight.size()
        self.mask.fill_(1)
        if mask_type == 'A': 
          self.mask[:,:,height//2, width//2:] = 0
          self.mask[:,:,height//2+1:,:] = 0
        if mask_type == 'B':
          self.mask[:,:,height//2,width//2+1:] = 0
          self.mask[:,:,height//2+1:,:] = 0


class PixelCNNResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.ModuleList([
            LayerNorm(dim),
            nn.ReLU(),
            MaskConv2d('B', dim, dim // 2, 1),
            LayerNorm(dim // 2),
            nn.ReLU(),
            MaskConv2d('B', dim // 2, dim // 2, 3, padding=1),
            LayerNorm(dim // 2),
            nn.ReLU(),
            MaskConv2d('B', dim // 2, dim, 1)
        ])

    def forward(self, x):
        out = x
        for layer in self.block:
            out = layer(out)
        return x + out


class PixelCNN(nn.Module):
    def __init__(self, input_shape, code_size, dim=256, n_layers=7):
        """
        PixelCNN model. 

        Inputs: 
        - input_shape: (H, W) of the height abd width of the input
        - code_size: dimention of embedding vector in the codebook
        - dim: number of filters
        - n_layers: number of repeated block
        """
        super().__init__()
        # Since the input is a 2D grid of discrete values, 
        # we'll have an input (learned) embedding layer to map the discrete values to embeddings 

        self.embedding = nn.Embedding(code_size, dim)
        model = nn.ModuleList([MaskConv2d('A', dim, dim, 7, padding=3),
                               LayerNorm(dim), nn.ReLU()])
        for _ in range(n_layers - 1):
            model.append(PixelCNNResBlock(dim))
        model.extend([LayerNorm(dim), nn.ReLU(), MaskConv2d('B', dim, 512, 1),
                      nn.ReLU(), MaskConv2d('B', 512, code_size, 1)])
        self.net = model
        self.input_shape = input_shape
        self.code_size = code_size

    def forward(self, x):
        out = self.embedding(x).permute(0, 3, 1, 2).contiguous()
        for layer in self.net:
            out = layer(out)
        return out

    def loss(self, x):
        # --------------- TODO: ------------------ 
        # Implement the loss function for PixelCNN
        # ----------------------------------------
        out = self.forward(x)
        loss = F.cross_entropy(out, x)
        return OrderedDict(loss=loss)

    def sample(self, n):
        # ------ TODO: sample from the model --------------
        # Instruction: 
        # Note that the generation process should proceed row by row and pixel by pixel. 
        # *hint: use torch.multinomial for sampling
        # -------------------------------------------------
        height = self.input_shape[0]
        width = self.input_shape[1]
        samples = torch.zeros((n, *self.input_shape), dtype=torch.int64).cuda()
        with torch.no_grad():
            for i in range(height):
                for j in range(width):
                    out = self.forward(samples)
                    probs = torch.softmax(out[:,:, i, j], dim = 1)
                    sample = torch.multinomial(probs, 1).squeeze().float()
                    samples[:, i, j] = sample
        return samples


class Quantize(nn.Module):
    """
    Vector quantisation. 

    Inputs: 
        - size: number of embedding vector in the codebook
        - code_dim: dimention of embedding vector in the codebook
    """
    def __init__(self, size, code_dim):
        super().__init__()
        # We use nn.Embedding to store embedding vectors. 
        self.embedding = nn.Embedding(size, code_dim)
        self.embedding.weight.data.uniform_(-1./size,1./size)

        self.code_dim = code_dim
        self.size = size

    def forward(self, z):
        # -------------------- TODO --------------------
        # Look at section 3.1 of the paper: Neural Discrete Representation Learning
        # , and finish the vector quantisation process. 
        # Note: we have taken care of the straight-through estimator, 
        #       note how we achieve it by using *detach* in PyTorch
        # 
        # 1. Compute the distance between every pair of latent embedding and output features of the encoder
        # 2. Get the encoder input by finding nearest latent embeddings (Eq. (1) and (2) in the paper)
        # 3. The encoding indices is the indice of the retrieved latent embeddings
        # ----------------------------------------------
        
        # z 128 256 8 8
        # e 128 256
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.code_dim)
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + torch.sum(self.embedding.weight**2, dim=1) - 2 * torch.matmul(z_flattened, self.embedding.weight.t())
        encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.size).to(torch.device("cuda"))
        encodings.scatter_(1, encoding_indices, 1)
        quantized = torch.matmul(encodings, self.embedding.weight).view(z.shape)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        z = z.permute(0, 3, 1, 2).contiguous()


        return quantized, (quantized - z).detach() + z, encoding_indices.view(-1, 8, 8)


class VQVAENet(nn.Module):
    def __init__(self, code_dim, code_size):
        """
        VQ-VAE model. 

        Inputs: 
        - code_dim: dimention of embedding vector in the codebook
        - code_size: number of embedding vector in the codebook

        ------- Instruction -------
        - Build a codebook follow the instructions in *Quantize* class
        ---------------------------
        """
        super().__init__()
        self.code_size = code_size
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 4, stride=2, padding=1),
            ResidualBlock(256),
            ResidualBlock(256),
        )

        self.codebook = Quantize(code_size, code_dim)

        self.decoder = nn.Sequential(
            ResidualBlock(256),
            ResidualBlock(256),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 3, 4, stride=2, padding=1),
            nn.Tanh(),
        )

    def encode_code(self, x):
        with torch.no_grad():
            x = 2 * x - 1
            z = self.encoder(x)
            indices = self.codebook(z)[2]
            return indices

    def decode_code(self, latents):
        with torch.no_grad():
            latents = self.codebook.embedding(latents).permute(0, 3, 1, 2).contiguous()
            return self.decoder(latents).permute(0, 2, 3, 1).cpu().numpy() * 0.5 + 0.5

    def forward(self, x):
        # x_tilde is the reconstructed images
        # diff1 and diff2 follow the last two terms of training objective (see Eq.(3) in the paper: Neural Discrete Representation Learning)
        # , where beta is set to 1.0

        x = 2 * x - 1
        z = self.encoder(x)
        e, e_st, _ = self.codebook(z)
        x_tilde = self.decoder(e_st)

        diff1 = torch.mean((z - e.detach()) ** 2)
        diff2 = torch.mean((e - z.detach()) ** 2)
        return x_tilde, diff1 + diff2

def vq_vae_loss(x_tilde, diff, x):
    # -------------- TODO --------------
    # finish the loss for VQ-VAE model
    # ----------------------------------
    x = 2 * x - 1
    recon_loss = F.mse_loss(x_tilde, x) 
    reg_loss = diff
    loss = recon_loss + reg_loss
    return OrderedDict(loss=loss, recon_loss=recon_loss, reg_loss=reg_loss)

