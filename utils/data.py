import csv
import os
import random
import glob
import scipy.signal
import scipy.io.wavfile
from PIL import Image, ImageEnhance

import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, MNIST
import torchvision.transforms as transforms

import numpy as np

from .helper import reset_seed
from .vis import tensor_to_image, show_samples


def _extract_tensors(dset, num=None, x_dtype=torch.float32):
    """
    Extract the data and labels from a CIFAR10 dataset object and convert them to
    tensors.

    Input:
    - dset: A torchvision.datasets.CIFAR10 object
    - num: Optional. If provided, the number of samples to keep.
    - x_dtype: Optional. data type of the input image

    Returns:
    - x: `x_dtype` tensor of shape (N, 3, 32, 32)
    - y: int64 tensor of shape (N,)
    """
    x = torch.tensor(dset.data, dtype=x_dtype).permute(0, 3, 1, 2).div_(255)
    y = torch.tensor(dset.targets, dtype=torch.int64)
    if num is not None:
        if num <= 0 or num > x.shape[0]:
            raise ValueError(
                "Invalid value num=%d; must be in the range [0, %d]" % (num, x.shape[0])
            )
        x = x[:num].clone()
        y = y[:num].clone()
    return x, y


def cifar10(num_train=None, num_test=None, x_dtype=torch.float32):
    """
    Return the CIFAR10 dataset, automatically downloading it if necessary.
    This function can also subsample the dataset.

    Inputs:
    - num_train: [Optional] How many samples to keep from the training set.
      If not provided, then keep the entire training set.
    - num_test: [Optional] How many samples to keep from the test set.
      If not provided, then keep the entire test set.
    - x_dtype: [Optional] Data type of the input image

    Returns:
    - x_train: `x_dtype` tensor of shape (num_train, 3, 32, 32)
    - y_train: int64 tensor of shape (num_train, 3, 32, 32)
    - x_test: `x_dtype` tensor of shape (num_test, 3, 32, 32)
    - y_test: int64 tensor of shape (num_test, 3, 32, 32)
    """
    download = not os.path.isdir("cifar-10-batches-py")
    dset_train = CIFAR10(root=".", download=download, train=True)
    dset_test = CIFAR10(root=".", train=False)

    x_train, y_train = _extract_tensors(dset_train, num_train, x_dtype)
    x_test, y_test = _extract_tensors(dset_test, num_test, x_dtype)

    return x_train, y_train, x_test, y_test


def preprocess_cifar10(show_examples=True, x_dtype='float32'): 
    """
    Download CIFAR10 dataset and extract data and labels. 

    Input: 
    - show_examples: If True, show 120 examples of CIFAR10 dataset
    - x_dtype: dtype of numpy array

    Returns: 
    - data_dict: a dictionary containing data for training and testing sets
    """
    download = not os.path.isdir("cifar-10-batches-py")
    dset_train = CIFAR10(root=".", download=download, train=True)
    dset_test = CIFAR10(root=".", train=False)

    x_train = (dset_train.data.transpose(0, 3, 1, 2) / 255.).astype(x_dtype)
    y_train = np.array(dset_train.targets, dtype='int64')
    x_test = (dset_test.data.transpose(0, 3, 1, 2) / 255.).astype(x_dtype)
    y_test = np.array(dset_test.targets, dtype='int64')
    # Visualize some examples from the dataset.
    if show_examples:
        classes = [
            "plane",
            "car",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]
        samples_per_class = 12
        samples = []
        reset_seed(0)
        for y, cls in enumerate(classes):
            plt.text(-4, 34 * y + 18, cls, ha="right")
            idxs = np.nonzero(y_train == y)[0]
            for i in range(samples_per_class):
                idx = idxs[random.randrange(idxs.shape[0])].item()
                samples.append(x_train[idx] * 255.)
        
        show_samples(samples, nrow=samples_per_class, title='CIFAR10 Samples')
    
    data_dict = {}
    data_dict["X_train"] = x_train
    data_dict["X_test"] = x_test

    return data_dict


def preprocess_mnist(show_examples=True, x_dtype='float32', binary_input=True): 
    """
    Download MNIST dataset and extract data and labels. 

    Input: 
    - show_examples: If True, show 128 examples of MNIST dataset
    - x_dtype: dtype of numpy array
    - binary_input: If True, binarize the image (0 or 1 for each pixel)

    Returns: 
    - data_dict: a dictionary containing data and labels for training and testing sets
    """
    download = not os.path.isdir("mnist-batches-py")
    dset_train = MNIST(root='.', download=download, train=True)
    dset_test = MNIST(root=".", train=False)

    # normalize the data to [0,1]
    x_train = (dset_train.data / 255.).numpy().astype(x_dtype)
    
    if binary_input: 
      thresh_cut = 0.2
      x_train[x_train > thresh_cut] = 1
      x_train[x_train < thresh_cut] = 0
      y_train = np.array(dset_train.targets, dtype='int64')
      
    x_test = (dset_test.data / 255.).numpy().astype(x_dtype)
    if binary_input: 
      x_test[x_test > thresh_cut] = 1
      x_test[x_test < thresh_cut] = 0
      y_test = np.array(dset_test.targets, dtype='int64')

    if show_examples:
        samples = []
        for i in range(128): 
            idx = random.randrange(x_train.shape[0])
            samples.append(x_train[idx] * 255.)
        show_samples(samples, nrow=10, title='MNIST Samples', gray=True)

    data_dict = {}
    data_dict["X_train"] = x_train
    data_dict["X_test"] = x_test
    data_dict['Y_train'] = y_train
    data_dict['Y_test'] = y_test
    
    return data_dict


class ImgAudDset(object): 
    def __init__(self, list_sample, split='train'): 
        """
        Data loader for audio-visual paired samples.
        It'll return paired audio and image. 

        Inputs: 
        - list_sample: a python list of the video paths
        - split: whether it's a train or validation dataset; choices: train/val
        """
        self.split = split
        self.augment = True
        self.img_size = 256
        self.vid_dur = 0.96 # length of the audio
        self.fps = 10 # fps of the video
        self.samp_sr = 16000  # sample rate of the audio

        if split == 'train': 
            vision_transform_list = [
                                    transforms.Resize((self.img_size, self.img_size)), 
                                    transforms.RandomHorizontalFlip(), 
                                    transforms.ToTensor()
                                    ]
        else: 
            vision_transform_list = [
                                    transforms.Resize((self.img_size, self.img_size)), 
                                    transforms.ToTensor()
                                    ]
        self.vision_transform = transforms.Compose(vision_transform_list)

        # load list of samples
        if isinstance(list_sample, str):
            self.list_sample = []
            csv_file = csv.reader(open(list_sample, 'r'), delimiter=' ')
            for row in csv_file:
                self.list_sample.append(row[0])
                
        if self.split == 'val': 
            random.seed(1234)
            np.random.seed(1234)

        if self.split == 'train': 
            random.shuffle(self.list_sample)
        num_sample = len(self.list_sample)

        # Set the frames and audio clip for every validation samples
        frames_per_sample = int(self.vid_dur * self.fps)
        if self.split in ['val', 'test']: 
            start_point = frames_per_sample
            self.val_rand = (start_point * np.ones(num_sample)).astype(int)

        print('# sample of {}: {}'.format(self.split, num_sample))

    def __getitem__(self, index): 
        sample = self.list_sample[index][4:]
        
        frame_folder = os.path.join(sample, 'frames')
        frame_list = glob.glob('%s/*.jpg' % frame_folder) # get all image samples
        frame_list.sort() 
        frames_per_sample = int(self.vid_dur * self.fps)  

        # load audio
        audio_path = os.path.join(sample, 'audio', 'audio.wav')
        audio_rate, audio = scipy.io.wavfile.read(audio_path) 
        audio = np.array(audio, 'double') / np.iinfo(audio.dtype).max
        if len(audio.shape) == 2: 
            audio = np.mean(audio, axis=-1)
        audio = scipy.signal.resample(audio, int(len(audio)/audio_rate*self.samp_sr))

        if self.split == 'train': 
            # make sure to randomly clip the audio in a valid range 
            total_length = min(len(frame_list), int(audio.shape[0] / self.samp_sr * self.fps))
            assert (total_length - frames_per_sample - 1) > 0, "video is {}, shape is {}, total:{}, frame per sample:{}".format(audio_path, audio.shape, total_length, frames_per_sample)
            rand_start = np.random.choice(total_length - frames_per_sample - 1)
        else: 
            rand_start = self.val_rand[index]

        frame_list = frame_list[rand_start: rand_start + frames_per_sample]

        if self.split == 'train': 
            frame_list = [random.choice(frame_list)]
        else: 
            frame_list = frame_list[frames_per_sample // 2 : frames_per_sample // 2 + 1]
        
        frame_info = frame_list[0]
        
        # clip and process audio data
        sample_len = int(self.vid_dur * self.samp_sr)
        total_length = int(audio.shape[0] / self.samp_sr * self.fps)
        assert (total_length - frames_per_sample - 1) > 0, "video is {}, shape is {}, total:{}, frame per sample:{}".format(audio_path, audio.shape, total_length, frames_per_sample)
        rand_start = np.random.choice(total_length - frames_per_sample - 1)
        audio_start = int(rand_start / self.fps * self.samp_sr)
        audio = audio[audio_start: audio_start + sample_len]
        assert audio.shape[0] == sample_len, "{} is broken".format(sample)
        audio = self.normalize_audio(audio)
        audio = torch.from_numpy(audio.copy()).float()
        vision_dict = self.process_video(frame_list)

        batch = {
            'info': audio_path,
            'frame_info': frame_info, 
            'frames': vision_dict['frameset'], 
            'audio': audio,
            'index': index
        }
        return batch

    def __len__(self): 
        return len(self.list_sample)

    def normalize_audio(self, samples, desired_rms=0.1, eps=1e-4):
        rms = np.maximum(eps, np.sqrt(np.mean(samples**2)))
        samples = samples * (desired_rms / rms)
        return samples

    def process_video(self, image_list):
        frame_list = []
        bright_random = random.random()
        color_random = random.random()
        for image in image_list:
            image = Image.open(image).convert('RGB')            
            if self.split == 'train' and self.augment:
                enhancer = ImageEnhance.Brightness(image)
                image = enhancer.enhance(bright_random*0.6 + 0.7)
                enhancer = ImageEnhance.Color(image)
                image = enhancer.enhance(color_random*0.6 + 0.7)            

            image = self.vision_transform(image)
            frame_list.append(image.unsqueeze(0))

        frame_list = torch.cat(frame_list, dim=0).squeeze()
        return {
            'frameset': frame_list, 
        }

