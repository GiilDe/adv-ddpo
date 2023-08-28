from dataclasses import dataclass
from datasets import load_dataset
from torchvision import transforms
import torch
from diffusers import DDPMPipeline
import torch.nn.functional as F
from diffusers.optimization import get_cosine_schedule_with_warmup

from PIL import Image
from diffusers import DDPMScheduler
from diffusers import UNet2DModel
from pretrain_model import MnistClassifier
from art.utils import load_mnist
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier
from ddpo_pytorch.rewards import l2_norm_diff, l_inf_norm_diff
import os


@dataclass
class TrainingConfig:
    train_batch_size = 16
    eval_batch_size = 16
    num_epochs = 1
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 1
    save_model_epochs = 30
    mixed_precision = "no"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "diffusion_testing"
    seed = 0
    l_inf_noise = 0


config = TrainingConfig()


config.dataset_name = "mnist"
dataset = load_dataset(config.dataset_name, split="train")


preprocess = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

to_tensor = transforms.ToTensor()
normalize = transforms.Normalize([0.5], [0.5])


def transform(examples):
    images = [to_tensor(image.convert("L")) for image in examples["image"]]
    return {"images": images}


dataset.set_transform(transform)


train_dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=config.train_batch_size, shuffle=True
)


pipeline_orig = DDPMPipeline.from_pretrained("nabdan/mnist_20_epoch")
model: UNet2DModel = pipeline_orig.unet


# noise_scheduler_ = DDPMScheduler(num_train_timesteps=1000)
noise_scheduler = DDPMScheduler.from_pretrained("nabdan/mnist_20_epoch")


optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
)


def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new("L", size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid


def evaluate(config, epoch, pipeline):
    images = pipeline(
        batch_size=config.eval_batch_size,
        generator=torch.manual_seed(config.seed),
    ).images

    image_grid = make_grid(images, rows=4, cols=4)

    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")


from accelerate import Accelerator
from tqdm.auto import tqdm
import os

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
_, _, min_pixel_value, max_pixel_value = load_mnist()
criterion = torch.nn.CrossEntropyLoss()
classifier = PyTorchClassifier(
    model=MnistClassifier(),
    clip_values=(min_pixel_value, max_pixel_value),
    loss=criterion,
    optimizer=optimizer,
    input_shape=image_shape,
    nb_classes=10,
)
attack = FastGradientMethod(
    estimator=classifier, eps=0.3, eps_step=0.1, batch_size=config.train_batch_size
)


def train_loop(
    config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler
):
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
    )
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers("train_example")

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0

    for epoch in range(config.num_epochs):
        progress_bar = tqdm(
            total=len(train_dataloader), disable=not accelerator.is_local_main_process
        )
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images = batch["images"]
            device = clean_images.device
            if config.l_inf_noise > 0:
                clean_images += torch.rand_like(clean_images) * config.l_inf_noise

            adversrial_images = attack.generate(clean_images.cpu().numpy())
            adversrial_images = torch.from_numpy(adversrial_images).to(device)

            adversrial_images = normalize(adversrial_images)

            noise = torch.randn(adversrial_images.shape).to(device)
            bs = adversrial_images.shape[0]

            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bs,),
                device=device,
            ).long()

            noisy_images = noise_scheduler.add_noise(
                adversrial_images, noise, timesteps
            )

            with accelerator.accumulate(model):
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

        if accelerator.is_main_process:
            pipeline = DDPMPipeline(
                unet=accelerator.unwrap_model(model), scheduler=noise_scheduler
            )

            if (
                epoch + 1
            ) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                evaluate(config, epoch, pipeline)

            if (
                epoch + 1
            ) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                pipeline.save_pretrained(config.output_dir)


from accelerate import notebook_launcher

args = (config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)

notebook_launcher(train_loop, args, num_processes=1)
