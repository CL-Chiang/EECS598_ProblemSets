import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.utils import save_image

import numpy as np

from tqdm import trange, tqdm_notebook

from .bigan_model import Generator, Discriminator, Encoder

class Solver(object):
    def __init__(self, train_data, test_data, n_epochs=100, batch_size=128, latent_dim=50):
        """
        Solver for training BiGAN and linear classifier on top of encoded features. 
        Inputs: 
        - train_data: a DataLoader for training samples
        - test_data: a DataLoader for testing samples
        - n_epochs: number of epochs to train BiGAN model. 
                    the total epochs to train linear classifier is n_epoch // 2
        - batch_size
        - latent_dim: dimention of random noise / embedding. 
        """
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.train_loader, self.test_loader = self.create_loaders(train_data, test_data)
        self.n_batches_in_epoch = len(self.train_loader)
        self.n_epochs = n_epochs
        self.curr_itr = 0

    def build(self):
        # --------------------------
        # Build BiGAN network, linear classifier and optimizers
        # 1. For generator and discriminator of BiGAN, 
        #    use Adam optimizer with learning rate = 2e-4, betas = (0.5, 0.999),
        #    and l2 weight decay=2.5e-5. 
        # 2. For discriminator of BiGAN, 
        #    use Adam optimizer with learning rate = 2e-4, and betas = (0, 0.9)  
        # 3. For BiGAN, schedule learning rate linearly to 0 over the whole training process
        # 4. Build linear classifier. 
        # 5. For linear classifier, use Adam optimizer with learning rate = 1e-3. 
        # ---------------------------

        self.d = Discriminator(self.latent_dim, 784).cuda()
        self.e = Encoder(784, self.latent_dim).cuda()
        self.g = Generator(self.latent_dim, 784).cuda() 

        self.g_optimizer = torch.optim.Adam(list(self.e.parameters()) + list(self.g.parameters()), 
                                            lr=2e-4, betas=(0.5, 0.999), weight_decay=2.5e-5)
        self.g_scheduler = torch.optim.lr_scheduler.LambdaLR(self.g_optimizer,
                                                             lambda epoch: (self.n_epochs - epoch) / self.n_epochs,
                                                             last_epoch=-1)
        self.d_optimizer = torch.optim.Adam(self.d.parameters(), lr=2e-4, betas=(0, 0.9))
        self.d_scheduler = torch.optim.lr_scheduler.LambdaLR(self.d_optimizer,
                                                             lambda epoch: (self.n_epochs - epoch) / self.n_epochs,
                                                             last_epoch=-1)
        self.linear = nn.Linear(self.latent_dim, 10).cuda()
        self.linear_optimizer = torch.optim.Adam(self.linear.parameters(), lr=1e-3)

    def reset_linear(self):
        self.linear = nn.Linear(self.latent_dim, 10).cuda()
        self.linear_optimizer = torch.optim.Adam(self.linear.parameters(), lr=1e-3)

    def create_loaders(self, train_data, test_data): 
        # We use torch.utils.data.DataLoader to build train_loader and test_loader
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=self.batch_size, shuffle=False)

        return train_loader, test_loader

    def train_bigan(self):
        """
        Return: 
        - train_lossses: numpy array containing averaged training loss per epoch. 
        """
        train_losses = []
        val_losses = []
        for epoch_i in tqdm_notebook(range(self.n_epochs), desc='Epoch'):
            epoch_i += 1

            self.d.train()
            self.g.train()
            self.e.train()
            self.batch_loss_history = []

            for batch_i, (x, y) in enumerate(tqdm_notebook(self.train_loader, desc='Batch', leave=False)):
                batch_i += 1
                self.curr_itr += 1
                x = x.cuda().float()
                # ---------------------- TODO ----------------------- # 
                # You'll implement the loss function for training BiGAN. 
                # Record the losses for discriminator in self.batch_loss_history. 
                # Hint: follow the loss function in Eq.(3) of the paper 
                # Jeff Donahue, et al. Adversarial Feature Learning
                # --------------------------------------------------- # 
                if x.shape[0] != self.batch_size:
                    continue

                # do a minibatch update for discriminator
                self.d_optimizer.zero_grad()
                
                # z: (size, dim) [-1,1]
                z = torch.autograd.Variable((torch.rand(self.batch_size, self.latent_dim) - 0.5) * 2).cuda()
                # encoder: 784 => 50
                e_x = self.e(x)
                # generator: 50 => (shape[0], 1 , 28, 28)
                g_z = self.g(z)
                x = x.view(x.shape[0],-1)
                g_z = g_z.view(g_z.shape[0], -1)

                d_e = self.d(e_x.detach(), x)
                d_g = self.d(z, g_z.detach())

                d_loss = -torch.mean(torch.log(d_e + 1e-8) + torch.log(1 - d_g + 1e-8)) # TODO: implement the loss for discriminator
                
                d_loss.backward(retain_graph=True)
                
                self.d_optimizer.step()

                # generator and encoder update
                self.g_optimizer.zero_grad()
                d_e = self.d(e_x, x)
                d_g = self.d(z, g_z)
                g_loss = -torch.mean(torch.log(d_g + 1e-8) + torch.log(1 - d_e + 1e-8)) # TODO: implement the loss for generator
                
                g_loss.backward()
                self.g_optimizer.step()

                self.batch_loss_history.append(d_loss.item())

            # step the learning rate
            self.g_scheduler.step()
            self.d_scheduler.step()
            epoch_loss = np.mean(self.batch_loss_history)
            train_losses.append(epoch_loss)

        train_losses = np.array(train_losses)

        return train_losses

    def train_linear_classifier(self):
        """
        Return: 
        - train_lossses: numpy array containing averaged training loss per epoch 
        - val_accs : numpy array containing averaged classification accuracy for testset per epoch 
        """
        train_losses = []
        val_accs = []
        for epoch_i in tqdm_notebook(range(self.n_epochs // 2), desc='Epoch'):
            epoch_i += 1

            self.e.eval()
            self.linear.train()
            self.batch_loss_history = []

            for batch_i, (x, y) in enumerate(tqdm_notebook(self.train_loader, desc='Batch', leave=False)):
                batch_i += 1
                self.curr_itr += 1
                x = x.cuda().float() 
                y = y.cuda()

                # calculate loss, take gradient step
                self.linear_optimizer.zero_grad()
                z = self.e(x).detach()
                pred = self.linear(z)
                linear_loss = F.cross_entropy(pred, y)
                linear_loss.backward()
                self.linear_optimizer.step()

                self.batch_loss_history.append(linear_loss.item())

            val_acc = self.val_acc()
            val_accs.append(val_acc)
            epoch_loss = np.mean(self.batch_loss_history)
            train_losses.append(epoch_loss)

        train_losses = np.array(train_losses)
        val_accs = np.array(val_accs)

        return train_losses, val_accs
    
    def sample(self, n):
        self.g.eval()
        with torch.no_grad():
            z = (torch.rand(n, self.latent_dim).cuda() - 0.5) * 2
            samples = self.g(z).reshape(-1, 1, 28, 28)
        return samples.cpu().numpy()

    def get_reconstructions(self, x):
        self.g.eval()
        self.e.eval()
        with torch.no_grad():
            z = self.e(x)
            recons = self.g(z).reshape(-1, 1, 28, 28)
        return recons.cpu().numpy()

    def val_acc(self):
        self.e.eval()
        self.linear.eval()

        val_acc_total = 0
        val_items = 0
        with torch.no_grad():
            for (inputs, labels) in self.test_loader:
                inputs = inputs.cuda().float()
                z = self.e(inputs)
                labels = labels.cuda()
                logits = self.linear(z)
                predictions = torch.argmax(logits, dim=1)
                num_correct = torch.sum(predictions == labels).float()
                val_acc_total += num_correct
                val_items += inputs.shape[0]

        return (val_acc_total / val_items).cpu().numpy()


def train_bigan(train_data, test_data): 
    """
    A function completing training, testing and sampling of the BiGAN model .  
    Input: 
    - train_data: an MNIST object for training data. 
      - train_data.data is a 60000x28x28 tensor that contains images
    - test_data: an MNIST object for testing data. 
      - test_data.data is a 10000x28x28 tensor that contains images
    return: 
    - bigan_losses: a numpy array of size (M,) containing losses during the 
          training of BiGAN (per mini-batch) 
    - samples: numpy array of shape (N, H, W, C), containing random samples 
          generated from the trained model. 
    - reconstructions: numpy array of shape (N, H, W, C), containing original 
          images and reconstructed images pairs 
    - bigan_train_losses: a numpy array of size (M,) containing losses of the 
          linear classifier using features extracted from BiGAN (per epoch) 
    - random_losses: a numpy array of size (M,) containing losses of the linear 
          classifier using features extracted from the randomly initialized network (per epoch) 
    
    -----------------
    
    To train a model, we'll 
    - Follow the instruction in the class *Solver* to construct a training solver. 
    - Train a linear classifier using a randomly initialized feature encoder. 
    - Train BiGAN model
    - Get random generated samples and reconstruction image pairs using the trained model
    - Train a linear classifier using the trained BiGAN feature encoder. 

    Hyperparameters: 
    - Batch size: 128
    - Adam optimizer 
        - Learning rate: 2e-4
        - Total epochs: 100
        - weight_decay: 2.5e-5
        - betas: (0.5, 0.99)  
    """

    solver = Solver(train_data, test_data, n_epochs=100)
    solver.build()

    # Training linear classifier on random encoder and get random encoder accuracy
    train_losses, val_accs = solver.train_linear_classifier()

    # Train BiGAN model
    bigan_losses = solver.train_bigan()
    samples = solver.sample(100).transpose(0, 2, 3, 1) * 0.5 + 0.5  # sample from BiGAN model
    train_images = train_data.data[:20].reshape(20, 1, 28, 28) / 255.0
    train_img_tensor = train_images.float().cuda() * 2 - 1
    recons = solver.get_reconstructions(train_img_tensor) * 0.5 + 0.5  # get original and reconstructed images pairs from the trained model
    reconstructions = np.concatenate([train_images, recons], axis=0).transpose(0, 2, 3, 1)

    # Training linear classifier on BiGAN encoder and get BiGAN encoder accuracy
    solver.reset_linear()
    bigan_train_losses, bigan_val_accs = solver.train_linear_classifier()

    print(f"Final BiGAN test linear accuracy: {bigan_val_accs[-1]}")
    print(f"Final random encoder test linear accuracy: {val_accs[-1]}")
    
    return bigan_losses, samples, reconstructions, bigan_train_losses, train_losses

