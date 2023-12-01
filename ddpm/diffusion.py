import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm

from .unet1d import Unet1d
from .unet2d import Unet2d


class Diffusion(object):
    def __init__(self, timesteps: int, model, device=None) -> None:
        self.timesteps = timesteps
        # Define beta schedule
        self.betas = self._linear_beta_schedule()

        # Pre-calculate different terms for closed form
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        
        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        # Define model
        self.model = model.to(self.device)
    
    
    def _get_index_from_list(self, vals: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        """ helper function to get index from list, considering batch dimension
        
        Args:
            vals (torch.Tensor): list of values
            t (torch.Tensor): timestep
            x_shape (torch.Size): shape of input image

        Returns:
            torch.Tensor: value at timestep t
        """
        batch_size = t.shape[0] # batch_size
        out = vals.gather(-1, t.cpu()) # (batch_size, 1)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
    
    
    def _linear_beta_schedule(self, start=0.0001, end=0.02) -> torch.Tensor:
        """ linear beta schedule
        Args:
            start (float, optional): beta at timestep 0. Defaults to 0.0001.
            end (float, optional): beta at last timestep. Defaults to 0.02.

        Returns:
            torch.Tensor: beta schedule
        """
        return torch.linspace(start, end, self.timesteps)
    
    
    def forward(self, x_0: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """ forward process of diffusion model
        Args:
            x_0 (torch.Tensor): input image
            t (torch.Tensor): timestep
            
        Returns:
            tuple[torch.Tensor, torch.Tensor]: noisy image and noise
        """
        noise = torch.randn_like(x_0).to(self.device)
        sqrt_alphas_cumprod_t = self._get_index_from_list(
            self.sqrt_alphas_cumprod, t, x_0.shape
        )
        sqrt_one_minus_alphas_cumprod_t = self._get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )
        # mean + variance
        return sqrt_alphas_cumprod_t.to(self.device) * x_0.to(self.device) \
            + sqrt_one_minus_alphas_cumprod_t.to(self.device) * noise.to(self.device), \
            noise.to(self.device)
            
            
    @torch.no_grad()
    def sample_timestep(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Calls the model to predict the noise in the image and returns 
        the denoised image. 
        Applies noise to this image, if we are not in the last step yet.
        """
        betas_t = self._get_index_from_list(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self._get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = self._get_index_from_list(self.sqrt_recip_alphas, t, x.shape)
        
        # Call model (current image - noise prediction)
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * self.model(x, t) / sqrt_one_minus_alphas_cumprod_t
        )
        posterior_variance_t = self._get_index_from_list(self.posterior_variance, t, x.shape)
        
        if t == 0:
            # As pointed out by Luis Pereira (see YouTube comment)
            # The t's are offset from the t's in the paper
            return model_mean
        else:
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
        
    @torch.no_grad()
    def sampling(self, x_T: torch.Tensor) -> torch.Tensor:
        """ sampling process of diffusion model
        Args:
            x_T (torch.Tensor): input image (gaussian noise)
            
        Returns:
            torch.Tensor: denoised image
        """
        x = x_T
        for i in tqdm(reversed(range(self.timesteps)), desc="Sampling"):
            t = torch.full((x.shape[0],), i, dtype=torch.long, device=self.device)
            x = self.sample_timestep(x, t)
        return x
    
    @torch.no_grad()
    def sampling_sequence(self, x_shape: torch.Size) -> np.ndarray:
        x_T = torch.randn(x_shape).to(self.device)
        sampled_tensor = self.sampling(x_T)
        sampled_seq = sampled_tensor.squeeze().detach().cpu().numpy()
        return sampled_seq
    
    @torch.no_grad()
    def sampling_image(self, x_shape: torch.Size) -> np.ndarray:
        x_T = torch.randn(x_shape).to(self.device)
        sampled_tensor = self.sampling(x_T)
        sampled_img = ((sampled_tensor + 1) / 2 * 255).squeeze().permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
        return sampled_img
        
        
if __name__ == "__main__":
    import cv2
    # img = cv2.imread("test.jpg")
    # # print(img.shape, img.max(), img.min())
    # # img = cv2.resize(img, (64, 64))
    # forward = Diffusion(300, device='cpu')
    # x0 = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0 * 2 - 1
    
    # num_images = 10
    # step_size = 300 // num_images
    # for idx in range(0, 300, step_size):
    #     # print(x0.shape, x0.max(), x0.min())
    #     t = torch.tensor([idx])
    #     x, noise = forward.forward(x0, t)
    #     img_noisy = ((x + 1) / 2 * 255).squeeze().permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
    #     # print(img_noisy.shape)
    #     cv2.imwrite(f"test_noisy{idx}.jpg", img_noisy)
    
    diffuser = Diffusion(300, device='cpu')
    img = diffuser.sampling_image(torch.Size([1, 3, 64, 64]))
    cv2.imwrite("sampled.jpg", img)