from CLIP.finetuner import Finetuner
from ddpm import Diffusion1d
from ddpm import Unet1d, Unet2d

model = Unet1d(dim=64, dim_mults=(1, 2, 4, 8), channels=1)
diffuser = Diffusion1d(time_steps=1000, sample_steps=10, model=model, device='cuda')
ft = Finetuner(batch_size=128, 
    epochs=100, 
    device="cuda", 
    diffuser=diffuser,
    clip_model_path="weights/pretrained_clip.pt",
)
ft.train()