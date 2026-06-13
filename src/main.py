import numpy as np
import time, pathlib, random
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as transforms
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim


SEED = 42


device = torch.device("mps")
torch.random.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


class Data(Dataset):
    def __init__(self, dir_name, transpose):
        inputs = np.load(dir_name + "/inputs.npy").transpose(0, 3, 1, 2)
        outputs = np.load(dir_name + "/outputs.npy")
        self.inputs = torch.from_numpy(inputs)
        self.outputs = torch.from_numpy(outputs)

        self.transpose = transpose
    
    def __len__(self):
        return len(self.outputs)
    
    def __getitem__(self, key):
        return self.transpose(self.inputs[key]), self.outputs[key]
    

transpose = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale = True),
    transforms.RandomRotation(degrees = 180),
    transforms.ColorJitter(brightness = 0.2, contrast = 0.2, saturation = 0.2),
    transforms.RandomHorizontalFlip(p = 0.5),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
data = Data("data", transpose)