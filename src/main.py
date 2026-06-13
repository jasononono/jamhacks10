import numpy as np
import random

import torch
from torchvision.transforms import v2

from consts import *
from dataset import Dataset
from network import CNN
from train import train


device = torch.device("mps")
torch.random.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


transpose = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale = True),
    v2.RandomRotation(degrees = 180),
    v2.ColorJitter(brightness = 0.2, contrast = 0.2, saturation = 0.2),
    v2.RandomHorizontalFlip(p = 0.5),
    v2.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
])
dataset = Dataset("data", device, transpose)


cnn = CNN().to(device)

train(cnn, dataset, 0)