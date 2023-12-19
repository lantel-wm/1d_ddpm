from ddpm import Trainer
from ddpm import Diffusion1d
from ddpm import Unet1d, Unet2d
from ddpm.dataloader import get_dataloader

from functools import partial
import torch
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    model_grids = torch.arange(1, 961)
    obs_grids = model_grids[model_grids % 4 == 0]
    H = torch.zeros(240, 960)
    for iobs in range(240):
        x1 = obs_grids[iobs] - 1
        H[iobs, x1] = 1.0
        
    model = Unet1d(dim=64, dim_mults=(1, 2, 4, 8), channels=1)
    # model = SimpleUnet()
    diffuser = Diffusion1d(time_steps=1000, sample_steps=10, model=model, device='cuda', model_path='weights/uncond_ddpm_1000.pt', H=H)
    
    data_path = '/mnt/ssd/L05_experiments_datasets/EnKF_F16_inf_1.08_loc_GC_before_DA_sz40_5y_2/data'
    analy = np.load(data_path + '/analy.npy')
    obs = np.load(data_path + '/zobs.npy')
    prior = np.load(data_path + '/prior.npy')
    
    plt.plot(prior[1145], label='prior')
    plt.plot(analy[1145], label='analy')
    plt.scatter(np.arange(0, 960, 4), obs[1145], c='r', s=1, marker='*', label='obs')
    plt.legend()
    plt.savefig('prior.png')
    plt.close()
    
    print(analy.shape, obs.shape, prior.shape)
    
    x_f = torch.Tensor(prior[1145, :]).unsqueeze(0).unsqueeze(0).float().to('cuda')
    x_a = torch.Tensor(analy[1145, :]).unsqueeze(0).unsqueeze(0).float().to('cuda')
    y_o = torch.Tensor(obs[1145, :]).unsqueeze(0).unsqueeze(0).float().to('cuda')
    
    
    
    x_guided_sample = diffuser.sampling_guided_sequence(x_f.shape, x_f, y_o, s_f=0, s_o=20)
    # x_sample = diffuser.sampling_sequence(x_f.shape)
    
    with open('datasets/image_min_max.txt', 'r') as f:
        min_val = float(f.readline())
        max_val = float(f.readline())
    # min_val = prior[1145].min()
    # max_val = prior[1145].max()
    
    x_guided_sample = 0.5 * (x_guided_sample + 1) * (max_val - min_val) + min_val
    # x_sample = 0.5 * (x_sample + 1) * (max_val - min_val) + min_val
    
    # plt.plot(x_sample, label='uncond')
    plt.plot(x_guided_sample, label='guided')
    plt.plot(prior[1145], label='prior')
    plt.scatter(np.arange(0, 960, 4), obs[1145], c='r', s=1, marker='*', label='obs')
    plt.xlim(0, 960)
    plt.title('observation guidance')
    plt.legend()
    plt.savefig('sample.png')
    plt.close()