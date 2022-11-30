from functools import wraps
from os import path, listdir
from timeit import default_timer

import pandas as pd
import torchvision.transforms as transforms

# NOTE: Global variables

ABS_PATH = path.dirname(path.realpath(__file__))
DATA_PATH = path.join(ABS_PATH, '../data')
IMG_PATH = path.join(ABS_PATH, '../img')
OUT_PATH = path.join(ABS_PATH, '../outputs')

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

# NOTE: Data transforms

sequence_transform = transforms.Compose([
    transforms.ToTensor(),
])

label_transform = transforms.Compose([
    transforms.ToTensor(),
])