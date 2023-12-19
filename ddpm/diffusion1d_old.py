import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm

from .unet1d import Unet1d
from .unet2d import Unet2d


class Diffusion1d(object):
    def __init__(self, time_steps: int, sample_steps: int, model, H=None, device=None, model_path=None) -> None:
        self.time_steps = time_steps
        self.sample_steps = sample_steps
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
        if model_path is not None:
            self.model.load_state_dict(torch.load(model_path))
        if H is not None:
            self.H = H.to(self.device) # forward operator, 240 x 960
    
    
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
        return torch.linspace(start, end, self.time_steps)
    
    
    def forward(self, x_0: torch.Tensor, t: torch.Tensor, type='forecast') -> tuple[torch.Tensor, torch.Tensor]:
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
            
    def forward_obs(self, x_0, y_o, t):
        """ forward process of diffusion model
        Args:
            x_0 (torch.Tensor): input image
            y_o (torch.Tensor): observation
            t (torch.Tensor): timestep
            
        Returns:
            tuple[torch.Tensor, torch.Tensor]: noisy image and noise
        """
        noise = torch.randn_like(x_0).to(self.device)
        noise = torch.matmul(noise, self.H.T)
        
        sqrt_alphas_cumprod_t = self._get_index_from_list(
            self.sqrt_alphas_cumprod, t, y_o.shape
        )
        sqrt_one_minus_alphas_cumprod_t = self._get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, y_o.shape
        )
        # mean + variance
        return sqrt_alphas_cumprod_t.to(self.device) * y_o.to(self.device) \
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
        for i in tqdm(reversed(range(self.time_steps)), desc="Sampling"):
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
    def gradient_F_observation(self, x, y_o, t):
        y_t, _ = self.forward_obs(x, y_o, t)
        return -2 * torch.matmul((torch.matmul(x, self.H.T) - y_t), self.H) / x.shape[-1]
        # return torch.matmul(y_t, self.H)
    
    @torch.no_grad()
    def gradient_F_forecast(self, x, x_f, t):
        x_ft, _ = self.forward(x_f, t)
        return -2 * (x - x_ft) / x.shape[-1]
        # return x_ft
    
    @torch.no_grad()
    def sample_timestep_guidance(self, x_f, y_o, s_f, s_o, x, t):
        betas_cur = self._get_index_from_list(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self._get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        posterior_variance = self._get_index_from_list(self.posterior_variance, t, x.shape)
        sqrt_recip_alphas_cur = self._get_index_from_list(self.sqrt_recip_alphas, t, x.shape)
        alphas_cumprod_cur = self._get_index_from_list(self.alphas_cumprod, t, x.shape)
        alphas_cumprod_prev = self._get_index_from_list(self.alphas_cumprod_prev, t, x.shape)
        
        dFo = self.gradient_F_observation(x, y_o, t) * s_o
        dFf = self.gradient_F_forecast(x, x_f, t) * s_f
        eps = self.model(x, t)

        sigma_t = 1.0 * ((1 - alphas_cumprod_prev) / (1 - alphas_cumprod_cur)) ** 0.5 \
            * ((1 - alphas_cumprod_cur / alphas_cumprod_prev)) ** 0.5
        # sigma_t = 0
        predicted_x0 = alphas_cumprod_prev ** 0.5 * (x - sqrt_one_minus_alphas_cumprod_t * eps) / (alphas_cumprod_cur ** 0.5)
        dir2_xt = (1 - alphas_cumprod_prev - sigma_t ** 2) ** 0.5 * eps
        
        noise = torch.randn_like(x) if t > 0 else 0
        
        return predicted_x0 + dir2_xt + posterior_variance * (dFo + dFf) + sigma_t * noise
    
    @torch.no_grad()
    def sampling_guided(self, x_T: torch.Tensor, x_f, y_o, s_f, s_o) -> torch.Tensor:
        """ sampling process of diffusion model
        Args:
            x_T (torch.Tensor): input image (gaussian noise)
            
        Returns:
            torch.Tensor: denoised image
        """
        x = x_T
        eta = 1
        ts = torch.linspace(self.time_steps, 0, (self.sample_steps + 1)).to(self.device).to(torch.long)
        # for i in tqdm(reversed(range(self.time_steps)), desc="Sampling"):
        #     t = torch.full((x.shape[0],), i, dtype=torch.long, device=self.device)
        #     x = self.sample_timestep_guidance(x_f, y_o, s_f, s_o, x, t)
        for i in tqdm(range(1, self.sample_steps + 1), desc='DDIM Sampling'):
            t = torch.full((x.shape[0],), ts[i], dtype=torch.long, device=self.device)
            x = self.sample_timestep_guidance(x_f, y_o, s_f, s_o, x, t)
        return x
    
    # DDIM
    @torch.no_grad()
    def sampling_guided_sequence(self, x_shape, x_f, y_o, s_f, s_o) -> np.ndarray:
        x_T = torch.randn(x_shape).to(self.device)
        sampled_tensor = self.sampling_guided(x_T, x_f, y_o, s_f, s_o)
        sampled_seq = sampled_tensor.squeeze().detach().cpu().numpy()
        return sampled_seq