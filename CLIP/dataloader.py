import os
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, random_split

# for CLIP training
class ImageTextPair(Dataset):
    def __init__(self, data_path, device="cpu"):
        self.data_path = data_path
        self.image = np.load(os.path.join(data_path, "zens_prior.npy"))
        self.text = np.load(os.path.join(data_path, "Hzens_prior.npy"))
        self.device = device
        self.preprocess()
        
    def preprocess(self):
        # min max normalization, scale between [-1, 1]
        image_max_val = self.image.max()
        image_min_val = self.image.min()
        text_max_val = self.text.max()
        text_min_val = self.text.min()
        
        with open(os.path.join(self.data_path, "image_min_max.txt"), "w") as f:
            f.write(f"{image_min_val}\n{image_max_val}")
        with open(os.path.join(self.data_path, "text_min_max.txt"), "w") as f:
            f.write(f"{text_min_val}\n{text_max_val}")
            
        if len(self.image.shape) == 3:
            self.image = self.image.reshape(-1, self.image.shape[-1])
        if len(self.text.shape) == 3:
            self.text = self.text.reshape(-1, self.text.shape[-1])
        # print(self.image.shape, self.text.shape)
        
        permutation = np.random.permutation(len(self.image))
        self.image = self.image[permutation]
        self.text = self.text[permutation]
        
        self.image = (self.image - image_min_val) / (image_max_val - image_min_val)
        self.image = self.image * 2 - 1
        self.image = torch.tensor(self.image).unsqueeze(1).float().to(self.device)
        
        self.text = (self.text - text_min_val) / (text_max_val - text_min_val)
        self.text = self.text * 2 - 1
        self.text = torch.tensor(self.text).unsqueeze(1).float().to(self.device)
        
    def __len__(self):
        return len(self.image)
    
    def __getitem__(self, idx):
        return self.image[idx], self.text[idx]
    
# for CLIP finetune
class ImageImagePair(Dataset):
    def __init__(self, data_path, device="cpu"):
        self.data_path = data_path
        self.image = np.load(os.path.join(data_path, "zens_prior.npy"))
        self.device = device
        self.preprocess()
        
    def preprocess(self):
        # min max normalization, scale between [-1, 1]
        with open(os.path.join(self.data_path, "image_min_max.txt"), "r") as f:
            min_val = float(f.readline())
            max_val = float(f.readline())
            
        if len(self.image.shape) == 3:
            self.image = self.image.reshape(-1, self.image.shape[-1])
        # print(self.image.shape, self.text.shape)
        
        permutation = np.random.permutation(len(self.image))
        self.image = self.image[permutation]
        
        self.image = (self.image - min_val) / (max_val - min_val)
        self.image = self.image * 2 - 1
        self.image = torch.tensor(self.image).unsqueeze(1).float().to(self.device)
        
    def __len__(self):
        return len(self.image)
    
    def __getitem__(self, idx):
        return self.image[idx]


def get_train_dataloader(batch_size: int, data_path: str, device='cpu') -> tuple[DataLoader, DataLoader]:
    data = ImageTextPair(data_path, device)
    train_size = int(0.8 * len(data))
    valid_size = len(data) - train_size
    trainset, validset = random_split(data, [train_size, valid_size])
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(validset, batch_size=batch_size, shuffle=False, drop_last=True)
    return train_loader, valid_loader

def get_finetune_dataloader(batch_size: int, data_path: str, device='cpu') -> tuple[DataLoader, DataLoader]:
    data = ImageImagePair(data_path, device)
    train_size = int(0.8 * len(data))
    valid_size = len(data) - train_size
    trainset, validset = random_split(data, [train_size, valid_size])
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(validset, batch_size=batch_size, shuffle=False, drop_last=True)
    return train_loader, valid_loader


def get_dataloader(batch_size: int, data_path: str, device='cpu', task="train") -> tuple[DataLoader, DataLoader]:
    assert task in ["train", "finetune"], "task must be either train or finetune"
    match task:
        case "train":
            return get_train_dataloader(batch_size, data_path, device)
        case "finetune":
            return get_finetune_dataloader(batch_size, data_path, device)


if __name__ == "__main__":
    train_loader, valid_loader = get_dataloader(64, "./datasets", device="cuda")
    for img, txt in train_loader:
        print(img.shape, txt.shape)
        print(img[0, 0, :15])
        print(txt[0, 0, :15])
        break