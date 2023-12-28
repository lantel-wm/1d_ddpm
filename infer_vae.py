import torch
import torch.nn as nn
import numpy as np

import matplotlib.pyplot as plt

from vae.vae1d_resnet import VAE1d

vae = VAE1d(z_dim=64).cuda()
# vae.load_state_dict(torch.load("./weights/vae64.pt"))
vae.load_state_dict(torch.load("./results_vae/model_2.pt"))

xens = np.load("./datasets/zens_prior.npy")
x = xens[2333, 7, :]
x = torch.tensor(x).unsqueeze(0).unsqueeze(1).float().cuda()

print(x.shape)
z = vae.encode(x)
x_recon = vae.decode(z)


plt.plot(x.squeeze().detach().cpu().numpy())
plt.plot(x_recon.squeeze().detach().cpu().numpy())
plt.savefig("./vae_recon.jpg")
plt.close()

plt.plot(z.squeeze().detach().cpu().numpy())
plt.savefig("./vae_z.jpg")
plt.close()