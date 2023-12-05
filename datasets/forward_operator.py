import numpy as np

model_grids = np.arange(1, 961)
obs_density = 4
obs_grids = model_grids[model_grids % obs_density == 0]

Hk = np.mat(np.zeros((240, 960)))
for iobs in range(240):
    x1 = obs_grids[iobs] - 1
    Hk[iobs, x1] = 1.0
    
zens = np.load("zens_prior.npy")
Hzens = np.zeros((zens.shape[0], zens.shape[1], 240))
for istep in range(zens.shape[0]):
    Hzens[istep] = (Hk * zens[istep].T).T
    
np.save("Hzens_prior.npy", Hzens)