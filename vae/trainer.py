import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from .vae1d_cnn import VAE1d
from .dataloader import get_dataloader

import matplotlib.pyplot as plt


class Trainer:
    def __init__(
        self, 
        batch_size: int, 
        epochs: int, 
        device=None,
        data_path=None,
        z_dim=10
        ) -> None:
        
        self.batch_size = batch_size
        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = VAE1d(z_dim=z_dim).to(self.device)
        self.model_save_dir = "results_vae"
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
            
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=1125, T_mult=2, eta_min=1e-5)
        self.epochs = epochs
        self.dataloader = get_dataloader(batch_size, data_path, device=self.device)
        
    def save_model_weight(self, epoch):
        torch.save(self.model.state_dict(), f'{self.model_save_dir}/model_{epoch}.pt')
        
    def get_loss(self, x):
        x_recon, mean, logvar = self.model(x)
        recon_loss = F.mse_loss(x_recon, x, reduction='mean')
        # recon_loss = nn.BCELoss(reduction='sum')(x_recon, x)
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        
        plt.plot(x[0].squeeze().detach().cpu().numpy(), label="original")
        plt.plot(x_recon[0].squeeze().detach().cpu().numpy(), label="recon")
        plt.legend()
        plt.savefig(f"{self.model_save_dir}/recon.jpg")
        plt.close()
        
        
        return recon_loss + kl_loss
        
    def train(self):
        for epoch in range(self.epochs):
            loop = tqdm(self.dataloader, desc=f"Epoch {epoch}")
            # losses = []
            for data in loop:
                self.optimizer.zero_grad()
                
                loss = self.get_loss(data)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                # losses.append(loss.item())
                loop.set_postfix(loss=loss.item(), lr=self.scheduler.get_last_lr()[0])
                
            self.save_model_weight(epoch)
            # self.save_sampled_image(epoch, torch.Size([1, 1, 960]))
            # self.save_sampled_sequence(epoch, torch.Size([1, 1, 960]))