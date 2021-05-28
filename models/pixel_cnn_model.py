import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskConv2d(nn.Conv2d):
  def __init__(self, mask_type, *args, **kwargs):
    """
    Masked convolutional layer. 

    Input: 
    - mask_type: "A" or "B" 
    """
    assert mask_type == 'A' or mask_type == 'B'
    super().__init__(*args, **kwargs)
    self.register_buffer('mask', torch.zeros_like(self.weight))
    self.create_mask(mask_type)

  def forward(self, input, cond=None):
    batch_size = input.shape[0]
    out = F.conv2d(input, self.weight * self.mask, self.bias, self.stride,
                  self.padding, self.dilation, self.groups)
    return out
  
  def create_mask(self, mask_type):
    # --------- TODO: Implement Mask for type A and B layer here ------- #
    _, _, height, width = self.weight.size()
    self.mask.fill_(1)
    if mask_type == 'A': 
      self.mask[:,:,height//2, width//2:] = 0
      self.mask[:,:,height//2+1:,:] = 0
    if mask_type == 'B':
      self.mask[:,:,height//2,width//2+1:] = 0
      self.mask[:,:,height//2+1:,:] = 0


class PixelCNN(nn.Module):
  def __init__(self, input_shape, n_colors, n_filters=64,
               kernel_size=7, n_layers=5):
    """
    Simple PixelCNN model. 
    
    Inputs: 
    - input_shape: size of input with shape (C, H, W)
    - n_colors: number of choices for every pixel
    - n_filters: number of filters for convolutional layers
    - kernel_size: size of kernel for convolutional layers
    - n_layers: number of masked type B convolutional layer with 7x7 kernel size and 64 output channels

    ------- Instruction -------
    We recommend the following network architecture: 
    - 1 masked type A convolutional layer with 7x7 kernel size and 64 output channels
    - 5 masked type B convolutional layer with 7x7 kernel size and 64 output channels
    - 2 masked type B convolutional layer with 1x1 kernel size and 64 output channels
    - A ReLU nonlinearities between every two convolutional layers

    You can start with constructing MaskConv2d object
    
    """
    super().__init__()
    n_channels = input_shape[0]
    
    block_init = lambda: MaskConv2d('B', n_filters, n_filters, 
                                    kernel_size=kernel_size,
                                    padding=kernel_size // 2)
    
    model = nn.ModuleList([MaskConv2d('A', n_channels, n_filters, 
                                      kernel_size=kernel_size,
                                      padding=kernel_size // 2)])
    for _ in range(n_layers):
      model.extend([nn.ReLU(), block_init()])
    model.extend([nn.ReLU(), MaskConv2d('B', n_filters, n_filters, 1)])
    model.extend([nn.ReLU(), MaskConv2d('B', n_filters, n_colors * n_channels, 1)])

    self.net = model
    self.input_shape = input_shape
    self.n_colors = n_colors
    self.n_channels = n_channels

  def forward(self, x):
    batch_size = x.shape[0]
    out = (x.float() / (self.n_colors - 1) - 0.5) / 0.5
    for layer in self.net:
        out = layer(out)
    return out.view(batch_size, self.n_colors, *self.input_shape)

  def sample(self, n):
    # ------ TODO: sample from the model. --------------
    # Instruction: 
    # Note that the generation process should proceed row by row and pixel by pixel. 
    # *hint: use torch.multinomial for sampling
    # --------------------------
    height = self.input_shape[1]
    width = self.input_shape[2]
    samples = torch.zeros(n, *self.input_shape).cuda()
    with torch.no_grad():
      for i in range(height):
        for j in range(width):
          for c in range(self.n_channels):
            out = self.forward(samples)
            probs = torch.softmax(out[:,:, c, i, j], dim = 1)
            sample = torch.multinomial(probs, 1).squeeze().float() / (self.n_colors - 1)
            samples[:, c, i, j] = sample
    return samples.cpu().numpy()


def pixel_cnn_loss(out, x):
  # --------- TODO: loss for training PixelCNN --------- 
  loss = F.cross_entropy(out, x.long())
  return loss


