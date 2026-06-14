import numpy as np
import random
import matplotlib.pyplot as plt

import torch
from torchvision.transforms import v2
from torch.utils.data import DataLoader, random_split

from classifier.consts import *
from classifier.dataset import Dataset
from classifier.network import CNN
import classifier.train as train


device = torch.device("mps" if torch.mps.is_available() else "cpu")
print("device:", device)

torch.random.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


transpose = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale = True),
    v2.RandomHorizontalFlip(p = 0.5),
    v2.ColorJitter(brightness = 0.2, contrast = 0.2, saturation = 0.2),
    v2.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
])
dataset = Dataset("data", device, transpose)
train_size = int(DATA_SPLIT * len(dataset))
test_size = len(dataset) - train_size
train_data, test_data = random_split(dataset, (train_size, test_size))

cnn = CNN().to(device)
try:
    history = train.train(cnn, device, train_data, EPOCHS)
except KeyboardInterrupt:
    pass

ans = input("save model? -> ")
if ans != "":
    torch.save({k: v.cpu() for k, v in cnn.state_dict().items()}, f"src/model/{ans}.pth")