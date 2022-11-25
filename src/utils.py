from functools import wraps
from os import path, listdir
from timeit import default_timer
from torchvision.io import read_image
from torch.utils.data import Dataset

import torch
import torchvision.transforms as transforms
import pandas as pd

# NOTE: Global variables

ABS_PATH = path.dirname(path.realpath(__file__))
DATA_PATH = path.join(ABS_PATH, '../data')
IMG_PATH = path.join(ABS_PATH, '../img')
OUT_PATH = path.join(ABS_PATH, '../outputs')

hidden_size = 5

params = {
    'hidden_size' : hidden_size,
    'hidenn_channels' : [1],
    'hidden_batch' : [False] * hidden_size,
    'hidden_dropout' : [False] * hidden_size,
    'hidden_dropout_per' : [0.25] * hidden_size,
    'hidden_conv_kernel_sizes' : [3] * hidden_size,
    'hidden_conv_strides' : [1] * hidden_size,
    'hidden_conv_padding' : [2] * hidden_size,
    'hidden_pool_kernel_sizes' : [2] * hidden_size,
    'hidden_pool_strides' : [2] * hidden_size,
    'num_classes' : 4,
    'fc_size' : 5,
}

# NOTE: Define types

types = {
    'COVID': 0,
    'LUNG_OPACITY': 1,
    'NORMAL': 2,
    'VIRAL_PNEUMONIA': 3
}

# NOTE: Transform from PIL to tensor

transform_pil = transforms.Compose([
    transforms.PILToTensor(),
])

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()
])

# NOTE: Decorator to time functions 

def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = default_timer()
        result = func(*args, **kwargs)
        end_time = default_timer()
        total_time = (end_time - start_time)
        print(f'Funct {func.__name__}, took {total_time:.4f} s')
        # print(f'{total_time:.4f}')

        return result

    return timeit_wrapper

# NOTE: Dataset class from Pytorch

class ImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None, device='cpu'):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.device = device

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label

# NOTE: Read dataset from path
@timeit
def read_dataset(device):
    X = []
    Y = []

    for type in types:
        print(f'Reading {type} - {types[type]}')
        p = path.join(DATA_PATH, type)
        tensors = path.join(p, 'tensors')

        for f in listdir(tensors):
            try:
                # Read tensor and send to device
                tensor = torch.load(path.join(tensors, f))
                X.append(tensor)
                Y.append(types[type])
            except:
                print(f'Error processing {path.join(tensors, f)}')
                break
        
        # NOTE: Tensors to device
        # Send list of tensors to device

        Y = torch.LongTensor(Y).to(device)

    return X, Y

@timeit
def to_device(X, Y, device):
    X = [x.to(device) for x in X]
    Y = torch.LongTensor(Y).to(device)

    return X, Y