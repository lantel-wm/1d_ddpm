import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm.auto import tqdm
from .clip import CLIP
from .dataloader import get_dataloader

class Trainer:
    def __init__(self, batch_size=64, epochs=100, device=None) -> None:
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = CLIP()
        
        if device is not None:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model.to(self.device)
        self.train_loader, self.valid_loader = get_dataloader(batch_size, "./datasets", device=self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=30000, eta_min=1e-7)
        self.save_dir = "results_clip"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
    
        
    def train(self):
        best_valid_loss = float('inf')
        for epoch in range(self.epochs):
            self.train_epoch(epoch)
            valid_loss = self.valid_epoch(epoch)
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(self.model.state_dict(), f"results_clip/best.pt")
            torch.save(self.model.state_dict(), f"results_clip/model_{epoch}.pt")
                
    def train_epoch(self, epoch):
        loop = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for img, txt in loop:
            loss = self.model(img, txt)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            loop.set_postfix(train_loss=loss.item(), lr=self.scheduler.get_last_lr()[0])
            
    def valid_epoch(self, epoch) -> float:
        loop = tqdm(self.valid_loader, desc=f"  Validating Epoch {epoch}")
        losses = []
        for img, txt in loop:
            loss = self.model(img, txt)
            loop.set_postfix(valid_loss=loss.item())
            losses.append(loss.item())
        
        return sum(losses) / len(losses)
                
