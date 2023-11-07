import numpy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor

class CustomDataset(Dataset):
    def __init__(self, np_X, np_y, normalize):
        if normalize:
            self.data = self.custom_norm_fun(np_X)
        else:
            self.data = torch.from_numpy(np_X).long()

        self.labels = np_y
        self.length = len(np_X)

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        sample = self.data[item]
        label = self.labels[item]
        return sample, label

    def custom_norm_fun(self, data):
        data = data/255
        return data
