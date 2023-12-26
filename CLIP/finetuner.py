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
            finetune_model_path=None,
            diffuser=None,
        ) -> None:
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = CLIPFinetune(clip_model_path=clip_model_path, device=device, diffusion_steps=1000)
        self.diffuser = diffuser
        
        if finetune_model_path is not None:
            self.model.load_state_dict(torch.load(finetune_model_path))
        
        if device is not None:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model.to(self.device)
        self.train_loader, self.valid_loader = get_dataloader(batch_size, 
                "./datasets", device=self.device, task="finetune")
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=30000, eta_min=1e-6)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=100, T_mult=2, eta_min=1e-6)
        data_size = len(self.train_loader)
        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[10 * data_size, 30 * data_size, 100 * data_size], gamma=0.1, last_epoch=-1)
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
        T = self.diffuser.time_steps
        t_range = [(1, 10), (10, 50), (50, 100), (100, T)]
        t_samples = [15, 15, 15, 15]
        Ts = [0, 99, 199, 299, 399, 499, 599, 699, 799, 899, 999]
        
        loop = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for img in loop:
            self.optimizer.zero_grad()
            # t = torch.randint(1, 300, (1,))
            timestep = random.choice(Ts)
            # timestep = random.choices(Ts, weights=[0.4, 0.1, 0.1, 0.1, 0.1, 0.05, 0.025, 0.025, 0.025, 0.025, 0.025], k=1)[0]
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
        Ts = [0, 99, 199, 299, 399, 499, 599, 699, 799, 899, 999]
        for time_step in Ts:
            loop = tqdm(self.valid_loader, desc=f"  Validating Epoch {epoch}")
            for img in loop:
                t = torch.tensor([time_step]).to(self.device)
                noisy_img, _ = self.diffuser.forward(img, t)
                loss = self.model(img, noisy_img, t)
                loop.set_postfix(valid_loss=loss.item(), t=t.item())
                losses.append(loss.item())
                
        print('\n')
        return sum(losses) / len(losses)
                
