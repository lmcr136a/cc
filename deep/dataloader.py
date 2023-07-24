import torch
from torch.utils.data import DataLoader, Dataset

import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import mplfinance as mpf
import time
import matplotlib.pyplot as plt
from scipy import io


class CHistoryDataset(Dataset):
    def __init__(self, train=True):
        if train:
            mat_file_name =  "train_dataset.mat"
        else:
            mat_file_name =  "test_dataset.mat"

        data = io.loadmat(mat_file_name)
        self.inputs = data['inputs']
        self.labels = data['labels']
        if len(self.labels) < 3:
            self.labels = self.labels[0]

        print(f"Input size: {self.inputs.shape}")
        print(f"label size: {len(self.labels)}")
        self.default_trans = transforms.ToTensor()

    def __getitem__(self, index):
        inputs = self.default_trans(self.inputs[index])
        return inputs, self.labels[index]
    
    def __len__(self):
        return len(self.inputs)
    

def get_dataloader(batch_size, train=False):
    coin_dataset = CHistoryDataset()
    dataloader = DataLoader(coin_dataset, batch_size=batch_size, shuffle=train)
    return dataloader