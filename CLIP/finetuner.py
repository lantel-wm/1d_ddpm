import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm.auto import tqdm
from copy import deepcopy
from .clip_finetune import CLIPFinetune
from .dataloader import get_dataloader

class Finetuner:
    def __init__(self, batch_size=64, 
            epochs=100, 
            device=None, 
            clip_model_path='weights/pretrained_clip.pt',
            diffuser=None,
        ) -> None:
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = CLIPFinetune(clip_model_path=clip_model_path, device=device)
        self.diffuser = diffuser
        
        
        if device is not None:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model.to(self.device)
        self.train_loader, self.valid_loader = get_dataloader(batch_size, 
                "./datasets", device=self.device, task="finetune")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=30000, eta_min=1e-7)
        self.save_dir = "results_finetune"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
    
        
    def train(self):
        best_valid_loss = float('inf')
        for epoch in range(self.epochs):
            # if epoch == 1:
            #     break
            self.train_epoch(epoch)
            valid_loss = self.valid_epoch(epoch)
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(self.model.state_dict(), f"{self.save_dir}/best.pt")
            torch.save(self.model.state_dict(), f"{self.save_dir}/model_{epoch}.pt")
    
        
    # @profile    
    def train_epoch(self, epoch):
        self.model.train()
        T = self.diffuser.timesteps
        loop = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for img in loop:
            self.optimizer.zero_grad()
            # t = torch.randint(1, 300, (1,))
            timestep = random.randint(1, T - 1)
            t = torch.ones((img.shape[0],)).to(self.device).long() * timestep
            
            noisy_img, _ = self.diffuser.forward(img, t)
            loss = self.model(img, noisy_img, timestep)
            
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            loop.set_postfix(train_loss=loss.item(), lr=self.scheduler.get_last_lr()[0], t=timestep)
            
        print('\n')
            
            
    def valid_epoch(self, epoch) -> float:
        self.model.eval()
        losses = []
        
        for time_step in [1, 10, 50, 100]:
            loop = tqdm(self.valid_loader, desc=f"  Validating Epoch {epoch}")
            for img in loop:
                t = torch.tensor([time_step]).to(self.device)
                noisy_img, _ = self.diffuser.forward(img, t)
                loss = self.model(img, noisy_img, t)
                loop.set_postfix(valid_loss=loss.item(), t=t.item())
                losses.append(loss.item())
                
        print('\n')
        return sum(losses) / len(losses)
                
