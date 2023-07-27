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

        # idxs = []
        # data['labels'] = data['labels'][0]
        # for i, d in enumerate(data['inputs']):
        #     if np.max(d) == np.min(d):
        #         idxs.append(i)
        #         print(i)
        # print(data['inputs'].shape, data['labels'].shape)
        # for i, idx in enumerate(idxs):
        #     data['inputs'] = np.delete(data['inputs'], int(idx-i), axis=0)
        #     data['labels'] = np.delete(data['labels'], int(idx-i), axis=0)
        # print(data['inputs'].shape, data['labels'].shape)
        # io.savemat(mat_file_name, data)

        self.inputs = data['inputs']
        self.labels = data['labels']
        if len(self.labels) < 3:
            self.labels = self.labels[0]
        print(f"Input size: {self.inputs.shape}")
        print(f"label size: {len(self.labels)}")
        self.default_trans = transforms.ToTensor()

    def __getitem__(self, index):
        
        inputs = minmax(self.inputs[index], index)
        inputs = self.default_trans(inputs)
        return inputs, self.labels[index]
    
    def __len__(self):
        return len(self.inputs)
    

def minmax(m, index):
    # print(m.shape, np.min(m), np.max(m))
    m = (m - np.min(m))/(np.max(m) - np.min(m))
    # 원래 0~1 사이인데, 그냥 -1~1로 하고싶음
    # m = 2*(m-0.5)
    return m

def get_dataloader(batch_size, train=False):
    coin_dataset = CHistoryDataset()
    dataloader = DataLoader(coin_dataset, batch_size=batch_size, shuffle=train)
    return dataloader