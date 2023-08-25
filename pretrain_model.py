## Training configuration

# For convenience, create a `TrainingConfig` class containing the training hyperparameters (feel free to adjust them):

from dataclasses import dataclass
import torch.nn.functional as F

from models import DDIMPipelineGivenImage, DDPMPipelineGivenImage
from diffusers import DDIMInverseScheduler, DDIMPipeline, DDIMScheduler
import torch
from PIL import Image
from diffusers import DDIMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers import DDPMPipeline
import os

from accelerate import Accelerator
from tqdm.auto import tqdm
import os
import torchvision

import torch
from ddpo_pytorch.rewards import l2_norm_diff, l_inf_norm_diff
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier
import torch.nn as nn
from art.utils import load_mnist, load_cifar10
from transformers.pipelines.image_classification import ImageClassificationPipeline
from transformers import pipeline, ViTImageProcessor


class CifarClassifier(nn.Module):
    def __init__(self, model, processor) -> None:
        super().__init__()
        self.model = model
        self.processor: ViTImageProcessor = processor

    def forward(self, images) -> torch.Tensor:
        tensor_images = self.processor.preprocess(images, return_tensors="pt")["pixel_values"].to("cuda")
        outputs = self.model(tensor_images)
        logits = outputs.logits
        return logits


pipe: ImageClassificationPipeline = pipeline(
    "image-classification", model="tzhao3/vit-CIFAR10"
)
classifier_ = pipe.model
classifier_ = CifarClassifier(classifier_, pipe.image_processor)


@dataclass
class TrainingConfig:
    train_batch_size = 32
    eval_batch_size = 32  # how many images to sample during evaluation
    num_epochs = 50
    gradient_accumulation_steps = 1
    learning_rate = 1e-5
    lr_warmup_steps = 500
    save_image_epochs = 1
    save_model_epochs = 30
    mixed_precision = "no"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "noisy-model-test"  # the model name locally and on the HF Hub
    seed = 0


num_steps = 10
# model_id = "nabdan/mnist_20_epoch"
model_id = "google/ddpm-cifar10-32"

cuda = torch.device("cuda")
pipeline_orig = DDIMPipelineGivenImage.from_pretrained(model_id).to(cuda)
pipeline_orig.scheduler = DDIMScheduler.from_config(pipeline_orig.scheduler.config)
# pipeline_orig.scheduler.set_timesteps(num_steps)

pipeline_train = DDIMPipelineGivenImage.from_pretrained(model_id).to(cuda)
pipeline_train.scheduler = DDIMScheduler.from_config(pipeline_train.scheduler.config)
# pipeline_noizy.scheduler.set_timesteps(num_steps)

config = TrainingConfig()
classifier_.model.to(cuda)

model = pipeline_train.unet

to_tensor = torchvision.transforms.ToTensor()
noise_scheduler = DDIMScheduler(num_train_timesteps=1000)
num_training_iterations = 10

if isinstance(pipeline_orig.unet.config.sample_size, int):
    image_shape = (
        config.train_batch_size,
        pipeline_orig.unet.config.in_channels,
        pipeline_orig.unet.config.sample_size,
        pipeline_orig.unet.config.sample_size,
    )
else:
    image_shape = (
        config.train_batch_size,
        pipeline_orig.unet.config.in_channels,
        *pipeline_orig.unet.config.sample_size,
    )

optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(num_training_iterations * config.num_epochs),
)

_, _, min_pixel_value, max_pixel_value = load_cifar10()
criterion = nn.CrossEntropyLoss()
classifier = PyTorchClassifier(
    model=classifier_,
    clip_values=(min_pixel_value, max_pixel_value),
    loss=criterion,
    optimizer=optimizer,
    input_shape=image_shape,
    nb_classes=10,
)


def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid


def evaluate(config, epoch, pipeline, postfix=None):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    images = pipeline(
        batch_size=config.eval_batch_size,
        generator=torch.manual_seed(config.seed),
        num_inference_steps=num_steps,
        eta=0.0,
    ).images

    # Make a grid out of the images
    image_grid = make_grid(images, rows=4, cols=4)

    # Save the images
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(
        f"{test_dir}/{epoch:04d}.png"
        if not postfix
        else f"{test_dir}/{epoch:04d}_{postfix}.png"
    )
    return images


original_images = evaluate(
    config,
    0,
    pipeline_orig,
    postfix="orig",
)


def evaluate_diff(images_noisy, original_images, config, epoch, device):
    l2_diff = l2_norm_diff(images_noisy, original_images, device=device)
    l_inf_diff = l_inf_norm_diff(images_noisy, original_images, device=device)
    print(f"l2_diff: {l2_diff}")
    print(f"l_inf_diff: {l_inf_diff}")
    print(f"l2_diff mean: {l2_diff.mean().item()}")
    print(f"l_inf_diff mean: {l_inf_diff.mean().item()}")


def train_loop(config, model, noise_scheduler, optimizer, lr_scheduler):
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        # log_with="wandb",
        project_dir=os.path.join(config.output_dir, "logs"),
    )
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers("train_example")

    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)

    global_step = 0

    attack = FastGradientMethod(
        estimator=classifier, eps=0.3, batch_size=config.train_batch_size
    )
    # img_prcoessor: ViTImageProcessor = pipe.image_processor

    for epoch in range(config.num_epochs):
        progress_bar = tqdm(
            total=num_training_iterations, disable=not accelerator.is_local_main_process
        )
        progress_bar.set_description(f"Epoch {epoch}")

        for step in range(num_training_iterations):
            noise = torch.randn(size=image_shape).to(cuda)
            clean_images = pipeline_orig(
                batch_size=config.train_batch_size,
                num_inference_steps=num_steps,
                image_=noise,
                eta=0.0,
            ).images
            clean_images = torch.stack([to_tensor(image) for image in clean_images]).to(
                cuda
            )
            # clean_images = 0.5 * torch.rand_like(clean_images) + clean_images

            # clean_images = img_prcoessor.preprocess(clean_images)
            adversrial_images = attack.generate(clean_images.cpu().numpy())
            adversrial_images = torch.from_numpy(adversrial_images).to(cuda)
            # Sample noise to add to the images
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bs,),
                device=clean_images.device,
            ).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(
                adversrial_images, noise, timesteps
            )[0]

            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "step": global_step,
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            if (
                epoch + 1
            ) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                images_noisy = evaluate(config, epoch, pipeline_train)
                evaluate_diff(
                    images_noisy, original_images, epoch, config, clean_images.device
                )

            # if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
            #     pipeline.save_pretrained(config.output_dir)


train_loop(config, model, noise_scheduler, optimizer, lr_scheduler)
