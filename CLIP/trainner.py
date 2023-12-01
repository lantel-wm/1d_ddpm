import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm.auto import tqdm
from resnet import ResNet, BasicBlock

class Trainner:
    def __init__(self, batch_size=64, epochs=100, img_length=960, txt_length=240, device=None) -> None:
        self.batch_size = batch_size
        self.epochs = epochs
        self.img_length = img_length
        self.txt_length = txt_length
        self.img_encoder = ResNet(img_channels=1, num_layers=18, block=BasicBlock) # resnet18
        self.txt_encoder = ResNet(img_channels=1, num_layers=18, block=BasicBlock) # resnet18
        
        if device is not None:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
        self.img_encoder.to(self.device)
        self.txt_encoder.to(self.device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        
        
    def train(self):
        for epoch in range(self.epochs):
            loop = tqdm(self.dataloader, desc=f"Epoch {epoch}")
            for data in loop:
                self.optimizer.zero_grad()
                
                img, txt = data
                loss = self.get_loss(img, txt)
                loss.backward()
                self.optimizer.step()
                loop.set_postfix(loss=loss.item())
                
