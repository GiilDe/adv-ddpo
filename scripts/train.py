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
from data.inversion_loader import load_inversion_data
from ddpo_pytorch.classification.factory import init_by_dataset
from ddpo_pytorch.inversion.ddim_inversion import DDIMInversion
import ddpo_pytorch.rewards
from ddpo_pytorch.stat_tracking import PerPromptStatTracker
from ddpo_pytorch.diffusers_patch.pipeline_with_logprob import pipeline_with_logprob
from ddpo_pytorch.diffusers_patch.ddim_with_logprob import ddim_step_with_logprob
import torch
import numpy as np
import wandb
from functools import partial
import tqdm
import torch.nn.functional as F
import torchvision
from models import DDIMPipelineGivenImage
from ddpo_pytorch.classification.factory import init_by_dataset
from utils import create_comparison_img, normalize_to_unit_interval, normalized_tensor_to_pil

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")

logger = get_logger(__name__)


def l_p_norm_loss(preds_a, preds_b, p=2):
    return torch.mean((preds_a - preds_b) ** p, dim=(1, 2, 3))


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

    # number of timesteps within each trajectory to train on
    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )

    accelerator = Accelerator(
        log_with="wandb" if config.log else None,
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        # we always accumulate gradients across timesteps; we want config.train.gradient_accumulation_steps to be the
        # number of *samples* we accumulate across, so we need to multiply by the number of training timesteps to get
        # the total number of optimizer steps to accumulate across.
        gradient_accumulation_steps=(
            config.train.gradient_accumulation_steps * (num_train_timesteps + 1)
        )
        if config.images_diff_weight_loss > 0  # i.e using diffusion loss
        else (config.train.gradient_accumulation_steps * num_train_timesteps),
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

    # load scheduler and models.
    pipeline_ft = DDIMPipelineGivenImage.from_pretrained(
        config.pretrained.pipeline_ft or config.pretrained.pipeline_original, revision=config.pretrained.revision
    )
    pipeline_orig = DDIMPipelineGivenImage.from_pretrained(
        config.pretrained.pipeline_original, revision=config.pretrained.revision
    )
    pipeline_orig = pipeline_orig.to(accelerator.device)
    pipeline_orig.unet.eval()
    pipeline_orig.unet.requires_grad_(False)
    # freeze parameters of models to save more memory
    pipeline_ft.unet.requires_grad_(not config.use_lora)
    # disable safety checker
    pipeline_ft.safety_checker = None
    # make the progress bar nicer
    pipeline_ft.set_progress_bar_config(
        position=1,
        disable=not accelerator.is_local_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )
    # switch to DDIM scheduler
    # pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", revision=config.pretrained.revision)
    pipeline_ft.scheduler.config[
        "steps_offset"
    ] = 1  # without adding this line there's a bug! the timestep = 0 returns logprob = nan which crashes the training
    pipeline_ft.scheduler = DDIMScheduler.from_config(pipeline_ft.scheduler.config)
    pipeline_orig.scheduler = DDIMScheduler.from_config(pipeline_ft.scheduler.config)
    inversion_pipe_ft = DDIMInversion(pipeline_ft.unet, pipeline_ft.scheduler, 300, pipeline_ft.progress_bar)
    inversion_pipe_org = DDIMInversion(pipeline_orig.unet, pipeline_orig.scheduler, 300, inversion_pipe_org.progress_bar)
    classifier = init_by_dataset(config.dataset)

    # pipeline_ft.scheduler.set_timesteps(1000)
    # pipeline_orig.scheduler.set_timesteps(1000)
    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to inference_dtype
    if config.use_lora:
        pipeline_ft.unet.to(accelerator.device, dtype=inference_dtype)

    if config.use_lora:
        # Set correct lora layers
        lora_attn_procs = {}
        for name in pipeline_ft.unet.attn_processors.keys():
            cross_attention_dim = (
                None
                if name.endswith("attn1.processor")
                else pipeline_ft.unet.config.cross_attention_dim
            )
            if name.startswith("mid_block"):
                hidden_size = pipeline_ft.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(
                    reversed(pipeline_ft.unet.config.block_out_channels)
                )[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = pipeline_ft.unet.config.block_out_channels[block_id]

            lora_attn_procs[name] = LoRAAttnProcessor(
                hidden_size=hidden_size, cross_attention_dim=cross_attention_dim
            )
        pipeline_ft.unet.set_attn_processor(lora_attn_procs)
        trainable_layers = AttnProcsLayers(pipeline_ft.unet.attn_processors)
    else:
        trainable_layers = pipeline_ft.unet

    # set up diffusers-friendly checkpoint saving with Accelerate

    def save_model_hook(models, weights, output_dir):
        assert len(models) == 1
        if config.use_lora and isinstance(models[0], AttnProcsLayers):
            pipeline_ft.unet.save_attn_procs(output_dir)
        elif not config.use_lora and isinstance(models[0], UNet2DConditionModel):
            models[0].save_pretrained(os.path.join(output_dir, "unet"))
        else:
            raise ValueError(f"Unknown model type {type(models[0])}")
        weights.pop()  # ensures that accelerate doesn't try to handle saving of the model

    def load_model_hook(models, input_dir):
        assert len(models) == 1
        if config.use_lora and isinstance(models[0], AttnProcsLayers):
            # pipeline.unet.load_attn_procs(input_dir)
            tmp_unet = UNet2DConditionModel.from_pretrained(
                config.pretrained.pipeline_ft,
                revision=config.pretrained.revision,
                subfolder="unet",
            )
            tmp_unet.load_attn_procs(input_dir)
            models[0].load_state_dict(
                AttnProcsLayers(tmp_unet.attn_processors).state_dict()
            )
            del tmp_unet
        elif not config.use_lora and isinstance(models[0], UNet2DConditionModel):
            load_model = UNet2DConditionModel.from_pretrained(
                input_dir, subfolder="unet"
            )
            models[0].register_to_config(**load_model.config)
            models[0].load_state_dict(load_model.state_dict())
            del load_model
        else:
            raise ValueError(f"Unknown model type {type(models[0])}")
        models.pop()  # ensures that accelerate doesn't try to handle loading of the model

    # accelerator.register_save_state_pre_hook(save_model_hook)
    # accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    to_tensor = torchvision.transforms.ToTensor()
    # Initialize the optimizer
    if config.train.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        trainable_layers.parameters(),
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    stat_tracker = PerPromptStatTracker(
        config.stat_tracking.buffer_size,
        config.stat_tracking.min_count,
    )

    # prepare reward fn
    classifier = init_by_dataset(config.dataset)
    reward_fn = getattr(ddpo_pytorch.rewards, config.reward_fn)(config, classifier)

    # for some reason, autocast is necessary for non-lora training but for lora training it isn't necessary and it uses
    # more memory
    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast

    # Prepare everything with our `accelerator`.
    trainable_layers, optimizer = accelerator.prepare(trainable_layers, optimizer)

    # executor to perform callbacks asynchronously. this is beneficial for the llava callbacks which makes a request to a
    # remote server running llava inference.
    executor = futures.ThreadPoolExecutor(max_workers=2)

    # Train!
    samples_per_epoch = (
        config.sample.batch_size
        * accelerator.num_processes
        * config.sample.num_batches_per_epoch
    )
    total_train_batch_size = (
        config.train.batch_size
        * accelerator.num_processes
        * config.train.gradient_accumulation_steps
    )

    if isinstance(pipeline_ft.unet.config.sample_size, int):
        image_shape = (
            config.sample.batch_size,
            pipeline_ft.unet.config.in_channels,
            pipeline_ft.unet.config.sample_size,
            pipeline_ft.unet.config.sample_size,
        )
    else:
        image_shape = (
            config.sample.batch_size,
            pipeline_ft.unet.config.in_channels,
            *pipeline_ft.unet.config.sample_size,
        )

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {config.num_epochs}")
    logger.info(f"  Sample batch size per device = {config.sample.batch_size}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}"
    )
    logger.info("")
    logger.info(f"  Total number of samples per epoch = {samples_per_epoch}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}"
    )
    logger.info(
        f"  Number of gradient updates per inner epoch = {samples_per_epoch // total_train_batch_size}"
    )
    logger.info(f"  Number of inner epochs = {config.train.num_inner_epochs}")
    logger.info(f"  Number of processes = {accelerator.num_processes}")

    assert config.sample.batch_size >= config.train.batch_size
    assert config.sample.batch_size % config.train.batch_size == 0
    assert samples_per_epoch % total_train_batch_size == 0

    if config.resume_from:
        logger.info(f"Resuming from {config.resume_from}")
        accelerator.load_state(config.resume_from)
        first_epoch = int(config.resume_from.split("_")[-1]) + 1
    else:
        first_epoch = 0

    global_step = 0
    for epoch in range(first_epoch, config.num_epochs):
        #################### SAMPLING ####################
        pipeline_ft.unet.eval()
        samples = []
        for i in tqdm(
            range(config.sample.num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            
            if epoch % config.evaluation_freq == 0 and accelerator.is_main_process:
                with torch.no_grad():
                    inversion_pipe_ft.model.eval()
                    inversion_pipe_org.model.eval()

                    l2_diff = []
                    l2_diff_recon = []
                    linf_diff = []
                    linf_diff_recon = []
                    img_acc = []
                    adv_acc = []
                    recon_acc = []

                    for i in range(config.num_eval_batches):
                        images, labels, latents = load_inversion_data(config.latents_dir, i)
                        adv_images = inversion_pipe_ft.ddim_loop(latents, is_forward=False, num_ddim_steps=config.sample.num_steps)
                        recon = inversion_pipe_org.ddim_loop(latents, is_forward=False, num_ddim_steps=config.sample.num_steps)

                        images = normalize_to_unit_interval(images)
                        adv_images = normalize_to_unit_interval(adv_images)
                        recon = normalize_to_unit_interval(recon)

                        diff = torch.flatten(images - adv_images, start_dim=1)
                        batch_l2_diff = torch.linalg.norm(diff, dim=1, ord=2)
                        batch_linf_diff = torch.linalg.norm(diff, dim=1, ord=float('inf'))
                        l2_diff = np.concatenate((l2_diff, batch_l2_diff.cpu().numpy()))
                        linf_diff = np.concatenate((linf_diff, batch_linf_diff.cpu().numpy()))

                        diff_recon = torch.flatten(images - recon, start_dim=1)
                        batch_l2_diff_recon = torch.linalg.norm(diff_recon, dim=1, ord=2)
                        batch_linf_diff_recon = torch.linalg.norm(diff_recon, dim=1, ord=float('inf'))
                        l2_diff_recon = np.concatenate((l2_diff_recon, batch_l2_diff_recon.cpu().numpy()))
                        linf_diff_recon = np.concatenate((linf_diff_recon, batch_linf_diff_recon.cpu().numpy()))

                        # # proccessed_img = classifier.precprocess_tensor(images).to(accelerator.device)
                        # proccessed_img = torch.stack( 
                        #     [classifier.preprocess(normalized_tensor_to_pil(image)[0]) for image in images]
                        # ).to(classifier.device)
                        # img_pred = classifier.predict(proccessed_img)
                        # img_acc_batch = (img_pred.argmax(dim=1) == labels).float().cpu().numpy()
                        # img_acc = np.concatenate((img_acc, img_acc_batch))

                        proccessed_adv = torch.stack( 
                            [classifier.preprocess(normalized_tensor_to_pil(image)[0]) for image in adv_images]
                        ).to(classifier.device)
                        adv_pred = classifier.predict(proccessed_adv).to(accelerator.device)
                        adv_acc_batch = (adv_pred.argmax(dim=1) == labels).float().cpu().numpy()
                        adv_acc = np.concatenate((adv_acc, adv_acc_batch))

                        proccessed_rec = torch.stack( 
                            [classifier.preprocess(normalized_tensor_to_pil(image)[0]) for image in recon]
                        ).to(classifier.device)
                        rec_pred = classifier.predict(proccessed_rec).to(accelerator.device)
                        rec_acc_batch = (rec_pred.argmax(dim=1) == labels).float().cpu().numpy()
                        recon_acc = np.concatenate((recon_acc, rec_acc_batch))

                    
                    # log rewards and images
                    if config.log:
                        accelerator.log(
                            {
                                "org_adv_l2": l2_diff.mean(),
                                "org_adv_linf": linf_diff.mean(),
                                "org_recon_l2": l2_diff_recon.mean(),
                                "org_recon_linf": linf_diff_recon.mean(),
                                # "benign_acc": img_acc.mean(),
                                "adv_acc": adv_acc.mean(),
                                "recon_acc": recon_acc.mean()
                            },
                            step=epoch,
                        )
                        accelerator.log(
                            {
                                "Original": [
                                    wandb.Image(images[j],
                                                caption=f"l2 diff: {batch_l2_diff[j]:.2f} \n l_inf diff: {batch_linf_diff[j]:.2f}")
                                    for j in range(0,16)
                                ],
                            },
                            step=epoch,
                        )
                        accelerator.log(
                            {
                                "Adverserial": [
                                    wandb.Image(adv_images[j],
                                                caption=f"l2 diff: {batch_l2_diff[j]:.2f} \n l_inf diff: {batch_linf_diff[j]:.2f}")
                                    for j in range(0,16)
                                ],
                            },
                            step=epoch,
                        )
                        accelerator.log(
                            {
                                "Recon": [
                                    wandb.Image(recon[j],
                                                caption=f"l2 diff: {batch_l2_diff_recon[j]:.2f} \n l_inf diff: {batch_linf_diff_recon[j]:.2f}")
                                    for j in range(0,16)
                                ],
                            },
                            step=epoch,
                        )
            # sample
            with autocast():
                (
                    images_adv,
                    latents,
                    log_probs,
                    all_variance_noize,
                ) = pipeline_with_logprob(
                    self=pipeline_ft,
                    noize=None,
                    num_inference_steps=config.sample.num_steps,
                    eta=config.sample.eta,
                    output_type="pil",
                    image_shape=image_shape,
                )

                noize = latents[0]

                (
                    images_orig,
                    latents_orig,
                    _,
                    _,
                    preds_orig,
                    timesteps_preds,
                ) = pipeline_with_logprob(
                    self=pipeline_orig,
                    noize=noize,
                    num_inference_steps=config.sample.num_steps,
                    eta=config.sample.eta,
                    output_type="pil",
                    image_shape=image_shape,
                    all_variance_noize=all_variance_noize,
                    return_preds=True,
                )

            latents = torch.stack(
                latents, dim=1
            )  # (batch_size, num_steps + 1, 4, 64, 64)
            log_probs = torch.stack(log_probs, dim=1)  # (batch_size, num_steps, 1)
            timesteps = pipeline_ft.scheduler.timesteps.repeat(
                config.sample.batch_size, 1
            )  # (batch_size, num_steps)

            # compute rewards asynchronously
            rewards = executor.submit(reward_fn, images_adv, images_orig, None)
            # yield to to make sure reward computation starts
            time.sleep(0)

            samples.append(
                {
                    "timesteps": timesteps,
                    "latents": latents[
                        :, :-1
                    ],  # each entry is the latent before timestep t
                    "next_latents": latents[
                        :, 1:
                    ],  # each entry is the latent after timestep t
                    "log_probs": log_probs,
                    "rewards": rewards,
                    "timesteps_preds": timesteps_preds,
                    "preds_orig": preds_orig,
                    "latents_orig": latents_orig,
                }
            )

        images_diffs_l2 = []
        images_diffs_l_inf = []
        accuracy = []
        ft_labels_scores = []
        # wait for all rewards to be computed
        for sample in tqdm(
            samples,
            desc="Waiting for rewards",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            rewards, reward_metadata = sample["rewards"].result()
            images_diffs_l2.append(reward_metadata["images_diff_l2"])
            images_diffs_l_inf.append(reward_metadata["images_diff_l_inf"])
            ft_labels_scores.append(reward_metadata["ft_labels_scores"])
            accuracy.append(reward_metadata["accuracy"])
            sample["rewards"] = torch.as_tensor(rewards, device=accelerator.device)

        images_diffs_l2 = torch.stack(images_diffs_l2)
        images_diffs_l_inf = torch.stack(images_diffs_l_inf)
        ft_labels_scores = torch.stack(ft_labels_scores)
        accuracy = torch.stack(accuracy).mean()
        # collate samples into dict where each entry has shape (num_batches_per_epoch * sample.batch_size, ...)
        samples = {k: torch.cat([s[k] for s in samples]) for k in samples[0].keys()}

        # gather rewards across processes
        rewards = accelerator.gather(samples["rewards"]).cpu().numpy()

        # log rewards and images
        if config.log:
            accelerator.log(
                {
                    "reward_histogram": rewards,
                    "labels_scores_histogram": ft_labels_scores.flatten(),
                    "epoch": epoch,
                    "reward_mean": rewards.mean(),
                    "reward_std": rewards.std(),
                    "images_diffs_l2_mean": images_diffs_l2.mean(),
                    "images_diffs_l_inf_mean": images_diffs_l_inf.mean(),
                    "images_diffs_l2_histgoram": images_diffs_l2.flatten(),
                    "images_diffs_l_inf_histgoram": images_diffs_l_inf.flatten(),
                    "accuracy_mean": accuracy,
                },
                step=global_step,
            )
            # this is a hack to force wandb to log the images as JPEGs instead of PNGs
            accelerator.log(
                {
                    "pertrubed images": [
                        wandb.Image(image, caption=f"{reward:.2f}")
                        for image, reward in zip(images_adv, rewards)
                    ],
                },
                step=global_step,
            )

            accelerator.log(
                {
                    "original images": [
                        wandb.Image(image, caption=f"{reward:.2f}")
                        for image, reward in zip(images_orig, rewards)
                    ],
                },
                step=global_step,
            )

        advantages = (
            stat_tracker.update(rewards)
            if config.historical_normalization
            else (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        )

        # ungather advantages; we only need to keep the entries corresponding to the samples on this process
        samples["advantages"] = (
            torch.as_tensor(advantages)
            .reshape(accelerator.num_processes, -1)[accelerator.process_index]
            .to(accelerator.device)
        )

        del samples["rewards"]

        total_batch_size, num_timesteps = samples["timesteps"].shape
        assert (
            total_batch_size
            == config.sample.batch_size * config.sample.num_batches_per_epoch
        )
        assert num_timesteps == config.sample.num_steps

        #################### TRAINING ####################
        for inner_epoch in range(config.train.num_inner_epochs):
            # shuffle samples along batch dimension
            perm = torch.randperm(total_batch_size, device=accelerator.device)
            samples = {k: v[perm] for k, v in samples.items()}

            # shuffle along time dimension independently for each sample
            perms = torch.stack(
                [
                    torch.randperm(num_timesteps, device=accelerator.device)
                    for _ in range(total_batch_size)
                ]
            )
            for key in ["timesteps", "latents", "next_latents", "log_probs"]:
                samples[key] = samples[key][
                    torch.arange(total_batch_size, device=accelerator.device)[:, None],
                    perms,
                ]

            # rebatch for training
            samples_batched = {
                k: v.reshape(-1, config.train.batch_size, *v.shape[1:])
                for k, v in samples.items()
            }

            # dict of lists -> list of dicts for easier iteration
            samples_batched = [
                dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
            ]

            # train
            pipeline_ft.unet.train()
            info = defaultdict(list)
            for i, sample in tqdm(
                list(enumerate(samples_batched)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=not accelerator.is_local_main_process,
            ):
                #################### TRAINING RL #################
                for j in tqdm(
                    range(num_train_timesteps),
                    desc="Timestep",
                    position=1,
                    leave=False,
                    disable=not accelerator.is_local_main_process,
                ):
                    with accelerator.accumulate(pipeline_ft.unet):
                        with autocast():
                            noise_pred = pipeline_ft.unet(
                                sample["latents"][:, j],
                                sample["timesteps"][:, j],
                            ).sample
                            # compute the log prob of next_latents given latents under the current model
                            _, log_prob, _ = ddim_step_with_logprob(
                                pipeline_ft.scheduler,
                                noise_pred,
                                sample["timesteps"][:, j],
                                sample["latents"][:, j],
                                eta=config.sample.eta,
                                prev_sample=sample["next_latents"][:, j],
                            )

                        # ppo logic
                        advantages = torch.clamp(
                            sample["advantages"],
                            -config.train.adv_clip_max,
                            config.train.adv_clip_max,
                        )
                        ratio = torch.exp(log_prob - sample["log_probs"][:, j])
                        unclipped_loss = -advantages * ratio
                        clipped_loss = -advantages * torch.clamp(
                            ratio,
                            1.0 - config.train.clip_range,
                            1.0 + config.train.clip_range,
                        )
                        loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))

                        # debugging values
                        # John Schulman says that (ratio - 1) - log(ratio) is a better
                        # estimator, but most existing code uses this so...
                        # http://joschu.net/blog/kl-approx.html
                        info["approx_kl"].append(
                            0.5
                            * torch.mean((log_prob - sample["log_probs"][:, j]) ** 2)
                        )
                        info["clipfrac"].append(
                            torch.mean(
                                (
                                    torch.abs(ratio - 1.0) > config.train.clip_range
                                ).float()
                            )
                        )
                        info["loss"].append(loss)

                        # backward pass
                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(
                                trainable_layers.parameters(),
                                config.train.max_grad_norm,
                            )
                        optimizer.step()
                        optimizer.zero_grad()

                #################### TRAINING DIFFUSION ##########
                if config.images_diff_weight_loss > 0:
                    timesteps_preds = sample["timesteps_preds"]
                    preds_orig = sample["preds_orig"]
                    latents_orig = sample["latents_orig"]

                    # noisy_images, sqrt_alpha_prod, sqrt_one_minus_alpha_prod = pipeline_ft.scheduler.add_noise(images_orig_deterministic, noize, timesteps)
                    # if config.normalize_threshold:
                    #     denominator = torch.linalg.vector_norm(noize - sqrt_one_minus_alpha_prod, ord=2, dim=(1, 2, 3))/torch.linalg.vector_norm(sqrt_alpha_prod, ord=2, dim=(1, 2, 3))
                    #     Cs = config.images_diff_threshold_loss/denominator
                    # else:
                    Cs = config.images_diff_threshold_loss
                    with accelerator.accumulate(pipeline_ft.unet):
                        # Predict the noise residual
                        noise_pred = pipeline_ft.unet(
                            latents_orig, timesteps_preds, return_dict=False
                        )[0]
                        loss = F.mse_loss(noise_pred, preds_orig)
                        diffusion_loss = ddpo_pytorch.rewards.hinge_loss(
                            loss, Cs, config.images_diff_weight_loss
                        )
                        info["diffusion_loss"].append(diffusion_loss)
                        accelerator.backward(diffusion_loss)

                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(
                                trainable_layers.parameters(),
                                config.train.max_grad_norm,
                            )
                        optimizer.step()
                        optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    assert (j == num_train_timesteps - 1) and (
                        i + 1
                    ) % config.train.gradient_accumulation_steps == 0
                    # log training-related stuff
                    info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                    info = accelerator.reduce(info, reduction="mean")
                    info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                    if config.log:
                        accelerator.log(info, step=global_step)
                    global_step += 1
                    info = defaultdict(list)

            # make sure we did an optimization step at the end of the inner epoch
            assert accelerator.sync_gradients

        if epoch != 0 and epoch % config.save_freq == 0 and accelerator.is_main_process:
            accelerator.save_state()


if __name__ == "__main__":
    app.run(main)
