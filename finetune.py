from CLIP.finetuner import Finetuner
from ddpm import Diffusion1d
from ddpm import Unet1d, Unet2d
import numpy as np
import torch

model = Unet1d(dim=64, dim_mults=(1, 2, 4, 8), channels=1)
diffuser = Diffusion1d(time_steps=1000, sample_steps=10, model=model, device='cuda')
ft = Finetuner(batch_size=128, 
    epochs=100, 
    device="cuda", 
    diffuser=diffuser,
    clip_model_path="weights/pretrained_clip.pt",
    # finetune_model_path="weights/clip_finetune_ddim.pt",
)
ft.train()

# data_path = '/mnt/ssd/L05_experiments_datasets/EnKF_F16_inf_1.08_loc_GC_before_DA_sz40_5y_2/data'
# analy = np.load(data_path + '/analy.npy')
# obs = np.load(data_path + '/zobs.npy')
# prior = np.load(data_path + '/prior.npy')

# x_f = torch.Tensor(prior[1145, :]).unsqueeze(0).unsqueeze(0).float().to('cuda')
# x_a = torch.Tensor(analy[1145, :]).unsqueeze(0).unsqueeze(0).float().to('cuda')
# y_o = torch.Tensor(obs[1145, :]).unsqueeze(0).unsqueeze(0).float().to('cuda')

# model_grids = torch.arange(1, 961)
# obs_grids = model_grids[model_grids % 4 == 0]
# H = torch.zeros(240, 960).to('cuda')
# for iobs in range(240):
#     x1 = obs_grids[iobs] - 1
#     H[iobs, x1] = 1.0
    
# hx_f = torch.matmul(H, x_f.squeeze(0).squeeze(0)).unsqueeze(0).unsqueeze(0)
# print(hx_f.shape)

# t = torch.Tensor([0]).to('cuda').long()
# x_f_noisy, _ = diffuser.forward(x_f, t)

# feature_noisy = ft.model.image_encoder(x_f_noisy, t)
# embd_noisy = ft.model.image_projection(feature_noisy)

# feature_img = ft.model.image_encoder(x_f)
# embd_img = ft.model.image_projection(feature_img)

# feature_text = ft.model.clip_model.text_encoder(hx_f)
# embd_text = ft.model.clip_model.text_projection(feature_text)

# print('img text', torch.dot(embd_img.squeeze(), embd_text.squeeze()))
# print('noisy img', torch.dot(embd_noisy.squeeze(), embd_img.squeeze()))
