import numpy as np
import random
import matplotlib.pyplot as plt

import torch
from torchvision.transforms import v2
from torch.utils.data import DataLoader, random_split

from consts import *
from dataset import Dataset
from network import CNN
import train


try:
    device = torch.device("mps")
except:
    device = torch.device("cps")
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
train_size = int(DATA_SPLIT * len(dataset))
test_size = len(dataset) - train_size
train_data, test_data = random_split(dataset, (train_size, test_size))

cnn = CNN().to(device)
history = train.train(cnn, device, train_data, EPOCHS)


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize = (8, 8))

ax1.set_title("loss history")
ax1.plot(history[0])
ax1.set_xlabel("epoch")
ax1.set_xticks(range(1, len(history[0]) + 1))
ax1.set_ylabel("loss")

ax2.set_title("accuracy history")
ax2.plot(history[1])
ax2.set_xlabel("epoch")
ax2.set_xticks(range(1, len(history[1]) + 1))
ax2.set_ylabel("loss")

ax3.set_title("training dataset")
print("Training Dataset")
train.test(cnn, device, train_data, ax3)
print("Testing Dataset")
ax4.set_title("testing dataset")
train.test(cnn, device, test_data, ax4)


plt.show()

ans = input("save model? -> ")
if ans != "":
    torch.save({k: v.cpu() for k, v in cnn.state_dict().items()}, f"src/model/{ans}.pth")