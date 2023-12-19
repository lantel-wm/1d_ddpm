import os
import cv2
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

# from .unet import Unet
from .dataloader import get_dataloader
# from .diffusion1d import Diffusion1d
from .utils import default


class Trainer:
    def __init__(
        self, 
        batch_size: int, 
        epochs: int, 
        diffuser,
        sampler,
        device=None
        ) -> None:
        
        self.batch_size = batch_size
        self.device = default(device, "cuda" if torch.cuda.is_available() else "cpu")
        # define diffusion model
        self.diffuser = diffuser
        self.T = self.diffuser.time_steps
        self.forward_diffusion_sample = self.diffuser.forward
        self.model = self.diffuser.model
        self.sampler = sampler
        self.model_save_dir = "results"
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
            
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=1125, T_mult=2, eta_min=1e-6)
        self.epochs = epochs
        self.dataloader = get_dataloader(batch_size, "./datasets", device=self.device)
    
    def get_loss(self, x_0, t):
        x_noisy, noise = self.forward_diffusion_sample(x_0, t)
        noise_pred = self.model(x_noisy, t)
        return F.l1_loss(noise, noise_pred)
    
    def save_model_weight(self, epoch):
        torch.save(self.model.state_dict(), f'{self.model_save_dir}/model_{epoch}.pt')
        
    def save_sampled_image(self, epoch, x_shape: torch.Size):
        sampled_img = self.sampler(x_shape, 'image')
        cv2.imwrite(f"{self.model_save_dir}/sampled_{epoch}.jpg", sampled_img)
        
    def save_sampled_sequence(self, epoch, x_shape: torch.Size):
        sampled_seq = self.sampler(x_shape)
        plt.plot(sampled_seq)
        plt.savefig(f"{self.model_save_dir}/sampled_seq_{epoch}.jpg")
        plt.close()
        
    def train(self):
        for epoch in range(self.epochs):
            loop = tqdm(self.dataloader, desc=f"Epoch {epoch}")
            # losses = []
            for data in loop:
                self.optimizer.zero_grad()
                
                # [0, T)
                t = torch.randint(0, self.T, (self.batch_size,)).to(self.device).long()
                data = data.to(self.device)
                loss = self.get_loss(data, t)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                # losses.append(loss.item())
                loop.set_postfix(loss=loss.item(), lr=self.scheduler.get_last_lr()[0])
                
            self.save_model_weight(epoch)
            # self.save_sampled_image(epoch, torch.Size([1, 1, 960]))
            self.save_sampled_sequence(epoch, torch.Size([1, 1, 960]))