import os
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt


class Lorenz05(Dataset):
    def __init__(self, data_path, device="cpu"):
        self.data_path = data_path
        self.data = np.load(os.path.join(data_path, "zens_prior.npy"))
        self.device = device
        self.preprocess()
        
    def preprocess(self):
        # min max normalization, scale between [-1, 1]
        with open(os.path.join(self.data_path, "min_max.txt"), "w") as f:
            f.write(f"{self.data.min()}\n{self.data.max()}")
        if len(self.data.shape) == 3:
            self.data = self.data.reshape(-1, 960)
        print(self.data.shape)
        self.data = self.data - self.data.min()
        self.data = self.data / self.data.max()
        self.data = self.data * 2 - 1
        self.data = torch.tensor(self.data).unsqueeze(1).float().to(self.device)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        return x


def get_dataloader(batch_size: int, data_path: str, device='cpu'):
    data = Lorenz05(data_path, device)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)
    return dataloader


if __name__ == "__main__":
    data_loader = get_dataloader(64, "./datasets", device="cuda")
    for data in data_loader:
        print(data.shape)
        print(data.max(), data.min())
        break