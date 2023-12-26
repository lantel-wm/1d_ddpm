import os
import torch
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
        # print(self.data.shape)
        self.data = self.data - self.data.min()
        self.data = self.data / self.data.max()
        self.data = self.data * 2 - 1
        self.data = torch.tensor(self.data).unsqueeze(1).float().to(self.device)
        
        model_grids = np.arange(1, 961)
        obs_density = 4
        obs_grids = model_grids[model_grids % obs_density == 0]

        Hk = torch.zeros((240, 960)).to(self.device)
        for iobs in range(240):
            x1 = obs_grids[iobs] - 1
            Hk[iobs, x1] = 1.0
        
        self.obs = torch.zeros((self.data.shape[0], 1, 240)).to(self.device)
        for istep in range(self.data.shape[0]):
            self.obs[istep, 0] = torch.matmul(Hk, self.data[istep].squeeze())
        self.obs = self.obs.float()
        
        # print(self.data.shape, self.obs.shape)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.obs[idx]


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