import numpy as np
import torch

from classifier.consts import *


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dir_name, device, transpose):
        inputs = np.load(dir_name + "/inputs.npy").transpose(0, 3, 1, 2)
        outputs = np.load(dir_name + "/outputs.npy")

        self.inputs = torch.from_numpy(inputs).to(device)
        self.outputs = torch.from_numpy(outputs).to(device)

        self.transpose = transpose
    
    def __len__(self):
        return len(self.outputs)
    
    def __getitem__(self, key):
        return self.transpose(self.inputs[key]), self.outputs[key].to(torch.float32)