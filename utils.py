import torch
from torch.utils.data import Dataset
import numpy as np

def split_data(data, window_size):
    length = len(data)
    split_data = []
    for i in range(0, length-window_size):
        split_data.append(data[i:i+window_size+1])
    return split_data

def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a,idx,axis=axis)


class LPDataset(Dataset):
    def __init__(self, path, window_size):
        super(LPDataset, self).__init__()
        self.data = torch.from_numpy(np.load(path))
        self.window_size = window_size
        self.num = self.data.size(0) - window_size

    def __len__(self):
        return self.num

    def __getitem__(self, item):
        return self.data[item: item + self.window_size], self.data[item + self.window_size]

class SDataset(Dataset):
    def __init__(self, path):
        self.data = torch.from_numpy(np.load(path).astype(np.float32))
        self.num = self.data.size(0)
        self.dimensions = self.data.size()

    def __len__(self):
        return self.num
 
    def __getitem__(self, item):
        return self.data[item][0:-1], self.data[item][-1]

    def data_dimensions(self):
        return list(self.dimensions)		 

def MSE(input, target):
    num = 1
    for s in input.size():
        num = num * s
    return (input - target).pow(2).sum().item() / num

def EdgeWiseKL(input, target):
    num = 1
    for s in input.size():
        num = num * s
    mask = (input > 0) & (target > 0)
    input = input.masked_select(mask)
    target = target.masked_select(mask)
    kl = (target * torch.log(target / input)).sum().item() / num
    return kl

def MissRate(input, target):
    num = 1
    for s in input.size():
        num = num * s
    mask1 = (input > 0) & (target == 0)
    mask2 = (input == 0) & (target > 0)
    mask = mask1 | mask2
    return mask.sum().item() / num
