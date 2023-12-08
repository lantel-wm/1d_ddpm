from CLIP.finetuner import Finetuner
from ddpm import Diffusion1d
from ddpm import Unet1d, Unet2d

model = Unet1d(dim=64, dim_mults=(1, 2, 4, 8), channels=1)
diffuser = Diffusion1d(timesteps=300, model=model, device='cuda')
ft = Finetuner(batch_size=64, 
    epochs=200, 
    device="cuda", 
    diffuser=diffuser,
    clip_model_path="weights/pretrained_clip.pt",
)
ft.train()