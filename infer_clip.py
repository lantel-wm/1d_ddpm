import os
import numpy as np
import torch
from CLIP.infer import CLIPInfer

device = "cuda"
clip = CLIPInfer("results_clip/best.pt", device=device)

data_path = '/mnt/ssd/L05_experiments_old/EnKF_F15_inf_1.01_before_DA_sz40_5y_cpptest/data'
# data_path = 'datasets'
zens = np.load(os.path.join(data_path, "zens_prior.npy"))
Hzens = np.load(os.path.join(data_path, "Hzens_prior.npy"))

with open(os.path.join('datasets', "min_max.txt"), "r") as f:
    min_val = float(f.readline())
    max_val = float(f.readline())

image = zens[5000, 10, :]
text = Hzens[10, 10, :]

image = (image - min_val) / (max_val - min_val)
image = image * 2 - 1
text = (text - min_val) / (max_val - min_val)
text = text * 2 - 1

image = torch.tensor(image).unsqueeze(0).unsqueeze(0).float().to(device)
text = torch.tensor(text).unsqueeze(0).unsqueeze(0).float().to(device)

print(image.shape, text.shape)

image_embedding = clip.image_encoder(image)
text_embedding = clip.text_encoder(text)

print(image_embedding.shape, text_embedding.shape)

print(torch.dot(image_embedding[0], text_embedding[0]))