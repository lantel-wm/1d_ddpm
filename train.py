import torch
from ddpm import Trainer
from ddpm import StableDiffusion1d
from ddpm import Unet1d, Unet2d
from ddpm.dataloader import get_dataloader
from vae.vae1d_resnet import VAE1d

from functools import partial

if __name__ == "__main__":
    # model = Unet(dim=64, dim_mults=(1, 2, 4, 8), flash_attn=True)'
    batch_size = 64
    unet = Unet1d(dim=64, dim_mults=(1, 2, 4, 8), channels=1)
    vae = VAE1d(z_dim=64, ch=8)
    # vae.load_state_dict(torch.load("./weights/vae64.pt"))
    # model = SimpleUnet()
    diffuser = StableDiffusion1d(time_steps=1000, sample_steps=1000, unet=unet, vae=vae, device='cuda')
    sampler = diffuser.sampling_sequence
    dataloader = get_dataloader(batch_size, "./datasets", device='cuda')
    trainer = Trainer(batch_size=batch_size, 
        epochs=20, 
        device="cuda", 
        diffuser=diffuser,
        sampler=sampler
    )
    trainer.train()