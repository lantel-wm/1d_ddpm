from ddpm import Trainer
from ddpm import Diffusion1d
from ddpm import Unet1d, Unet2d
from ddpm.dataloader import get_dataloader

from functools import partial

if __name__ == "__main__":
    # model = Unet(dim=64, dim_mults=(1, 2, 4, 8), flash_attn=True)'
    batch_size = 64
    model = Unet1d(dim=64, dim_mults=(1, 2, 4, 8), channels=1)
    # model = SimpleUnet()
    diffuser = Diffusion1d(time_steps=1000, sample_steps=1000, model=model, device='cuda')
    sampler = diffuser.sampling_sequence
    dataloader = get_dataloader(batch_size, "./datasets", device='cuda')
    trainer = Trainer(batch_size=batch_size, 
        epochs=100, 
        device="cuda", 
        diffuser=diffuser,
        sampler=sampler
    )
    trainer.train()