from collections import defaultdict
import contextlib
import os
import datetime
from concurrent import futures
import time
from absl import app, flags
from ml_collections import config_flags
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from diffusers import DDIMScheduler, UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
import numpy as np
from data.factory import init_by_dataset
from data.inversion_loader import save_inversion_data
from ddpo_pytorch.inversion.ddim_inversion_pipeline import DDIMInversionPipeline
from ddpo_pytorch.stat_tracking import PerPromptStatTracker
from ddpo_pytorch.diffusers_patch.pipeline_with_logprob import pipeline_with_logprob
from ddpo_pytorch.diffusers_patch.ddim_with_logprob import ddim_step_with_logprob
import torch
import wandb
from functools import partial
import tqdm
import torch.nn.functional as F
import torchvision
from utils import create_comparison_img, normalize_to_unit_interval, numpy_to_pil
from ddpo_pytorch.inversion.ddim_inversion import DDIMInversion

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")

logger = get_logger(__name__)

def main(_):
    # basic Accelerate and logging setup
    config = FLAGS.config

    if not config.run_name:
        config.run_name = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")

    if config.resume_from:
        config.resume_from = os.path.normpath(os.path.expanduser(config.resume_from))
        if "checkpoint_" not in os.path.basename(config.resume_from):
            # get the most recent checkpoint in this directory
            checkpoints = list(
                filter(lambda x: "checkpoint_" in x, os.listdir(config.resume_from))
            )
            if len(checkpoints) == 0:
                raise ValueError(f"No checkpoints found in {config.resume_from}")
            config.resume_from = os.path.join(
                config.resume_from,
                sorted(checkpoints, key=lambda x: int(x.split("_")[-1]))[-1],
            )

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )

    accelerator = Accelerator(
        log_with="wandb" if config.log else None,
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config
    )
    if config.log and accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="ddpo-pytorch",
            config=config.to_dict(),
            init_kwargs={"wandb": {"name": config.run_name}},
        )
    logger.info(f"\n{config}")

    # set seed (device_specific is very important to get different prompts on different devices)
    set_seed(config.seed, device_specific=True)

    repo_id = "google/ddpm-cifar10-32"
    pipeline = DDIMInversionPipeline.from_pretrained(
      repo_id, revision="main"
    )
    scheduler = DDIMScheduler.from_pretrained(repo_id)
    pipeline.scheduler = scheduler
    pipeline = pipeline.to("cuda")
    pipeline.unet.eval()
    pipeline.unet.requires_grad_(False)
    # freeze parameters of models to save more memory
    # make the progress bar nicer
    pipeline.set_progress_bar_config(
        position=1,
        # disable=not accelerator.is_local_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )
    inversion_pipe = DDIMInversion(pipeline.unet, scheduler, config.num_inversion_steps, pipeline.progress_bar)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16


    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # for some reason, autocast is necessary for non-lora training but for lora training it isn't necessary and it uses
    # more memory
    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast

    if isinstance(pipeline.unet.config.sample_size, int):
        image_shape = (
            config.sample.batch_size,
            pipeline.unet.config.in_channels,
            pipeline.unet.config.sample_size,
            pipeline.unet.config.sample_size,
        )
    else:
        image_shape = (config.sample.batch_size, pipeline.unet.config.in_channels, *pipeline.unet.config.sample_size)

    logger.info("***** Running Inversion *****")
    logger.info(f"  Sample batch size per device = {config.sample.batch_size}")
    logger.info(f"  Number of processes = {accelerator.num_processes}")

    #### data processing #######
    testset = init_by_dataset(config.dataset, config.sample.batch_size, root=config.data_cache, shuffle=False, train=False)
    data_loader = testset.dataloader

    for i, (images, labels) in enumerate(
        tqdm(
            data_loader,
            disable=not accelerator.is_local_main_process,
            position=0,
        )
    ):
        # sample
        with autocast():
            images = images.to(accelerator.device)
            labels = labels.to(accelerator.device)
            
            latents = inversion_pipe.invert(images, config.num_inversion_steps)

            if config.save_latents == True:
                save_inversion_data(images, labels, latents, config.latent_save_dir, i)

            recon = inversion_pipe.ddim_loop(latents, is_forward=False, num_ddim_steps=config.sample.num_steps, return_all=False)
        
        normalized_input = normalize_to_unit_interval(images)
        recon = normalize_to_unit_interval(recon)
        diff = torch.flatten(normalized_input - recon, start_dim=1)
        l2_diff = torch.linalg.norm(diff, dim=1, ord=2)
        linf_diff = torch.linalg.norm(diff, dim=1, ord=float('inf'))

        # log rewards and images
        if config.log:
            accelerator.log(
                {
                    "l2_recon_error": l2_diff.mean(),
                    "linf_recon_err": linf_diff.mean(),
                },
                step=i,
            )
            # this is a hack to force wandb to log the images as JPEGs instead of PNGs
            img_comparisons = create_comparison_img(normalized_input[:4], recon[:4])
            
            accelerator.log(
                {
                    "Inversion Reconstruction": [
                        wandb.Image(img_comparisons[j], caption=f"l2 diff: {l2_diff[j]:.2f} \n l_inf diff: {linf_diff[j]:.2f}")
                        for j in range(4)
                    ],
                },
                step=i,
            )
            
            # inv_process_img = numpy_to_pil(torchvision.utils.make_grid(normalize_to_unit_interval(latents[::30, 0])).cpu().permute(1, 2, 0).numpy())
            # backwards_latents = numpy_to_pil(torchvision.utils.make_grid(normalize_to_unit_interval(recon[::30, 0])).cpu().permute(1, 2, 0).numpy())
            # accelerator.log(
            #     {
            #         "Inversion Pipeline": 
            #             wandb.Image(inv_process_img[0], caption=f"jumps of 30")
            #     },
            #     step=i,
            # )
            # accelerator.log(
            #     {
            #         "Backwards Pipeline": 
            #             wandb.Image(backwards_latents[0], caption=f"jumps of 30")
            #     },
            #     step=i,
            # )


if __name__ == "__main__":
    app.run(main)
