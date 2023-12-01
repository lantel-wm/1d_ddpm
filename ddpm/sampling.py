# import torch
# import torch.nn as nn

# from diffusion import Diffusion

# class Sampler(Diffusion):
#     def __init__(self) -> None:
#         pass
    
#     @torch.no_grad()
#     def sample_timestep(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
#         """
#         Calls the model to predict the noise in the image and returns 
#         the denoised image. 
#         Applies noise to this image, if we are not in the last step yet.
#         """
#         betas_t = self._get_index_from_list(self.betas, t, x.shape)
#         sqrt_one_minus_alphas_cumprod_t = self._get_index_from_list(
#             self.sqrt_one_minus_alphas_cumprod, t, x.shape
#         )
#         sqrt_recip_alphas_t = self._get_index_from_list(self.sqrt_recip_alphas, t, x.shape)
        
#         # Call model (current image - noise prediction)
#         model_mean = sqrt_recip_alphas_t * (
#             x - betas_t * self.model(x, t) / sqrt_one_minus_alphas_cumprod_t
#         )
#         posterior_variance_t = self._get_index_from_list(self.posterior_variance, t, x.shape)
        
#         if t == 0:
#             # As pointed out by Luis Pereira (see YouTube comment)
#             # The t's are offset from the t's in the paper
#             return model_mean
#         else:
#             noise = torch.randn_like(x)
#             return model_mean + torch.sqrt(posterior_variance_t) * noise
        