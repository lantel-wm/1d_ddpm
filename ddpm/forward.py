# import torch
# import torch.nn.functional as F
# import numpy as np

# from diffusion import Diffusion

# class ForwardProcess(Diffusion):
    
#     def __init__(self, timesteps: int) -> None:
#         super().__init__(timesteps)

#     def forward(self, x_0, t, device="cpu"):
#         """ 
#         Takes an image and a timestep as input and 
#         returns the noisy version of it
#         """
#         noise = torch.randn_like(x_0)
#         sqrt_alphas_cumprod_t = self._get_index_from_list(
#             self.sqrt_alphas_cumprod, t, x_0.shape
#         )
#         sqrt_one_minus_alphas_cumprod_t = self._get_index_from_list(
#             self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
#         )
#         # mean + variance
#         return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
#             + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), \
#             noise.to(device)
        
        
# if __name__ == "__main__":
#     import cv2
#     img = cv2.imread("test.jpg")
#     # print(img.shape, img.max(), img.min())
#     # img = cv2.resize(img, (64, 64))
#     forward = ForwardProcess(300)
#     x0 = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0 * 2 - 1
    
#     num_images = 10
#     step_size = 300 // num_images
#     for idx in range(0, 300, step_size):
#         # print(x0.shape, x0.max(), x0.min())
#         t = torch.tensor([idx])
#         x, noise = forward.forward(x0, t)
#         img_noisy = ((x + 1) / 2 * 255).squeeze().permute(1, 2, 0).detach().numpy().astype(np.uint8)
#         # print(img_noisy.shape)
#         cv2.imwrite(f"test_noisy{idx}.jpg", img_noisy)

