from typing import Optional, Union, Tuple, List, Callable, Dict
from tqdm.notebook import tqdm
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch.nn.functional as nnf
import numpy as np
import abc
import shutil
from torch.optim.adam import Adam
from PIL import Image


class DDIMInversion:
    
    def prev_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        return prev_sample
    
    def next_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        timestep, next_timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample
    
    def get_noise_pred_single(self, latents, t):
        noise_pred = self.model(latents, t).sample
        return noise_pred

    def get_noise_pred(self, latents, t, is_forward=True, context=None):
        latents_input = latents
        if context is None:
            context = self.context
        noise_pred = self.model(latents_input, t).sample
        if is_forward:
            latents = self.next_step(noise_pred, t, latents)
        else:
            latents = self.prev_step(noise_pred, t, latents)
        return latents


    @torch.no_grad()
    def ddim_loop(self, latent, num_ddim_steps, is_forward=True, return_all=False):
        with self.progress_bar(total=num_ddim_steps) as progress_bar:
          self.schd.set_timesteps(num_ddim_steps)
          if return_all == True:
              all_latent = [latent]
          
          latent = latent.clone().detach()
          for i in range(num_ddim_steps):
              t_index = len(self.scheduler.timesteps) - i - 1 if is_forward else i
              t = self.scheduler.timesteps[t_index]
              noise_pred = self.get_noise_pred_single(latent, t)
              if is_forward:
                latent = self.next_step(noise_pred, t, latent)
              else:
                latent = self.prev_step(noise_pred, t, latent)

              if return_all == True:
                  all_latent.append(latent)
              
              progress_bar.update()


          if return_all == True:
              return all_latent
          return latent

    @property
    def scheduler(self):
        return self.schd

    @torch.no_grad()
    def ddim_inversion(self, image, num_steps):
        ddim_latents = self.ddim_loop(image, num_ddim_steps=num_steps)
        return ddim_latents

    
    def invert(self, image_gt,num_steps, num_inner_steps=10, early_stop_epsilon=1e-5, verbose=False):
        ddim_latents = self.ddim_inversion(image_gt, num_steps)
        return ddim_latents
        
    
    def __init__(self, model, scheduler, num_ddim_steps, progress_bar = None):
        self.schd = scheduler
        self.model = model
        self.progress_bar = progress_bar
        self.schd.set_timesteps(num_ddim_steps)