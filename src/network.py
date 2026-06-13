import torch
from torch import nn
import torch.nn.functional as f

from consts import *


class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        # (3, 128, 128)
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = 3, padding = 1) # (16, 128, 128)
        self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2) # (16, 64, 64)

        self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, padding = 1) # (32, 64, 64)
        self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2) # (32, 32, 32)

        self.conv3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, padding = 1) # (64, 32, 32)
        self.pool3 = nn.MaxPool2d(kernel_size = 2, stride = 2) # (64, 16, 16)

        self.lin1 = nn.Linear(64 * 16 * 16, 256)
        self.dropout = nn.Dropout(0.5)
        self.lin2 = nn.Linear(256, CLASSES)

    def forward(self, x):
        x = self.pool1(f.relu(self.conv1(x)))
        x = self.pool2(f.relu(self.conv2(x)))
        x = self.pool3(f.relu(self.conv3(x)))

        x = torch.flatten(x, 1)
        x = f.relu(self.lin1(x))
        x = self.dropout(x)
        x = self.lin2(x)

        return x