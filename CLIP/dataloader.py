import os
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, random_split

class ImageTextPair(Dataset):
    def __init__(self, data_path, device="cpu"):
        self.data_path = data_path
        self.image = np.load(os.path.join(data_path, "zens_prior.npy"))
        self.text = np.load(os.path.join(data_path, "Hzens_prior.npy"))
        self.device = device
        self.preprocess()
        
    def preprocess(self):
        # min max normalization, scale between [-1, 1]
        max_val = self.image.max()
        min_val = self.image.min()
        with open(os.path.join(self.data_path, "min_max.txt"), "w") as f:
            f.write(f"{min_val}\n{max_val}")
            
        if len(self.image.shape) == 3:
            self.image = self.image.reshape(-1, self.image.shape[-1])
        if len(self.text.shape) == 3:
            self.text = self.text.reshape(-1, self.text.shape[-1])
        # print(self.image.shape, self.text.shape)
        
        permutation = np.random.permutation(len(self.image))
        self.image = self.image[permutation]
        self.text = self.text[permutation]
        
        self.image = (self.image - min_val) / (max_val - min_val)
        self.image = self.image * 2 - 1
        self.image = torch.tensor(self.image).unsqueeze(1).float().to(self.device)
        
        self.text = (self.text - min_val) / (max_val - min_val)
        self.text = self.text * 2 - 1
        self.text = torch.tensor(self.text).unsqueeze(1).float().to(self.device)
        
    def __len__(self):
        return len(self.image)
    
    def __getitem__(self, idx):
        return self.image[idx], self.text[idx]


def get_dataloader(batch_size: int, data_path: str, device='cpu'):
    data = ImageTextPair(data_path, device)
    train_size = int(0.8 * len(data))
    valid_size = len(data) - train_size
    trainset, validset = random_split(data, [train_size, valid_size])
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(validset, batch_size=batch_size, shuffle=False, drop_last=True)
    return train_loader, valid_loader

    
if __name__ == "__main__":
    train_loader, valid_loader = get_dataloader(64, "./datasets", device="cuda")
    for img, txt in train_loader:
        print(img.shape, txt.shape)
        print(img[0, 0, :15])
        print(txt[0, 0, :15])
        break