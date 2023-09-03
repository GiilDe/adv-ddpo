from dataclasses import dataclass, asdict
from datasets import load_dataset
from torchvision import transforms
import torch
from diffusers import DDPMPipeline, DDIMPipeline
import torch.nn.functional as F
from diffusers.optimization import get_cosine_schedule_with_warmup

from PIL import Image
from diffusers import DDPMScheduler, DDIMScheduler
from diffusers import UNet2DModel
from pretraining_utils import (
    MnistClassifier,
    PipelineIterator,
    evaluate_diff,
    evaluate_diff_algo,
)
from art.utils import load_mnist
from art.attacks.evasion import FastGradientMethod, ZooAttack
from art.estimators.classification import PyTorchClassifier
import os
import logging
import wandb
from accelerate import Accelerator
from tqdm.auto import tqdm
from ddpo_pytorch.rewards import hinge_loss

logging.basicConfig(format="%(message)s", level=logging.INFO)


@dataclass
class TrainingConfig:
    train_batch_size: int
    eval_batch_size: int
    num_epochs: int
    gradient_accumulation_steps: int
    learning_rate: float
    lr_warmup_steps: int
    save_image_epochs: int
    save_model_epochs: int
    mixed_precision: str  # `no` for float32, `fp16` for automatic mixed precision
    output_dir: str
    seed: int
    l_inf_noise: float
    add_pertruebation: bool
    num_inference_steps: int
    diffusion_class: object
    scheduler_class: object
    save_images_to_disk: bool
    run_name: str
    training_steps: int
    data_from_model: bool
    train_clean_model: bool
    regularize_using_clean_model: bool
    reg_lambda: float
    hinge_loss_threshold: float
    data_length: int
    labels_from_data: bool

    def to_dict(self) -> dict:
        return {key: str(val) for key, val in asdict(self).items()}


config = TrainingConfig(
    train_batch_size=16,
    eval_batch_size=16,
    num_epochs=1,
    gradient_accumulation_steps=1,
    learning_rate=1e-4,
    lr_warmup_steps=500,
    save_image_epochs=1,
    save_model_epochs=30,
    mixed_precision="no",  # `no` for float32, `fp16` for automatic mixed precision,
    output_dir="pretraining_output",
    seed=0,
    l_inf_noise=0,
    add_pertruebation=True,
    num_inference_steps=2,
    diffusion_class=DDIMPipeline,
    scheduler_class=DDIMScheduler,
    save_images_to_disk=False,
    run_name="ZOO test",
    training_steps=1000,
    data_from_model=False,
    train_clean_model=False,
    regularize_using_clean_model=True,
    reg_lambda=3,
    hinge_loss_threshold=0.0,
    data_length=3,
    labels_from_data=True,
)

model_id = "nabdan/mnist_20_epoch"

config.dataset_name = "mnist"
dataset = load_dataset(config.dataset_name, split="train")

model = DDIMPipeline.from_pretrained(model_id).unet

noise_scheduler = config.scheduler_class.from_pretrained(
    model_id, num_train_timesteps=config.training_steps
)
noise_scheduler.set_timesteps(config.num_inference_steps)

model_clean = DDIMPipeline.from_pretrained(model_id).unet

to_tensor = transforms.ToTensor()
normalize = transforms.Normalize([0.5], [0.5])


def transform(examples):
    images = [to_tensor(image.convert("L")) for image in examples["image"]]
    return {"images": images, "labels": examples["label"]}


dataset.set_transform(transform)
dataset = dataset.select(range(config.data_length))
train_dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=config.train_batch_size, shuffle=True
)
if config.data_from_model:
    train_dataloader = PipelineIterator(
        DDIMPipeline(unet=model_clean, scheduler=noise_scheduler),
        config.train_batch_size,
        length=500,
    )


optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
)
if config.train_clean_model:
    optimizer_clean = torch.optim.AdamW(
        model_clean.parameters(), lr=config.learning_rate
    )
    lr_scheduler_clean = get_cosine_schedule_with_warmup(
        optimizer=optimizer_clean,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * config.num_epochs),
    )


def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new("L", size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid


def to_pil(images):
    images = images.cpu().permute(0, 2, 3, 1).numpy()
    images = DDIMPipeline.numpy_to_pil(images)
    return images


def evaluate(config, epoch, pipeline, postfix=""):
    logging.info(f"Generating samples for epoch {epoch}, {postfix}")
    logging.info(f"num inference steps: {pipeline.scheduler.num_inference_steps}")
    images = pipeline(
        batch_size=config.eval_batch_size,
        generator=torch.manual_seed(config.seed),
        num_inference_steps=config.num_inference_steps,
    ).images

    if config.save_images_to_disk:
        image_grid = make_grid(images, rows=4, cols=4)

        test_dir = os.path.join(config.output_dir, f"samples")
        os.makedirs(test_dir, exist_ok=True)
        image_grid.save(
            f"{test_dir}/{epoch:04d}.png"
            if not postfix
            else f"{test_dir}/{epoch:04d}_{postfix}.png"
        )
    return images


if isinstance(model.config.sample_size, int):
    image_shape = (
        config.train_batch_size,
        model.config.in_channels,
        model.config.sample_size,
        model.config.sample_size,
    )
else:
    image_shape = (
        config.train_batch_size,
        model.config.in_channels,
        *model.config.sample_size,
    )

(x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()
criterion = torch.nn.CrossEntropyLoss()
classifier = MnistClassifier()
classifier_attack = PyTorchClassifier(
    model=classifier,
    clip_values=(min_pixel_value, max_pixel_value),
    loss=criterion,
    optimizer=optimizer,
    input_shape=image_shape[1:],
    nb_classes=10,
    channels_first=True,
)
# attack = FastGradientMethod(
#     estimator=classifier_attack,
#     eps=0.3,
#     eps_step=0.1,
#     batch_size=config.train_batch_size,
# )
attack = ZooAttack(
    classifier=classifier_attack,
    confidence=0.0,
    targeted=False,
    # learning_rate=1e-1,
    max_iter=1,
    # binary_search_steps=10,
    # initial_const=1e-3,
    # abort_early=True,
    # use_resize=False,
    # use_importance=False,
    # nb_parallel=5,
    # batch_size=1,
    # variable_h=0.01,
)


original_images = evaluate(
    config,
    0,
    DDIMPipeline(unet=model_clean, scheduler=noise_scheduler),
    postfix="orig",
)


def train_loop(
    config,
    model,
    noise_scheduler,
    optimizer,
    train_dataloader,
    lr_scheduler,
    model_clean=None,
    optimizer_clean=None,
    lr_scheduler_clean=None,
):
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        # log_with="wandb",
        project_dir=os.path.join(config.output_dir, "logs"),
    )
    accelerator.init_trackers(
        project_name="ddpo-pytorch-pretraining",
        config=config.to_dict(),
        init_kwargs={"wandb": {"name": config.run_name}},
    )
    if config.output_dir is not None:
        os.makedirs(config.output_dir, exist_ok=True)

    global_step = 0

    accelerator.log(
        {
            "original images": [wandb.Image(image) for image in original_images],
        },
        step=global_step,
    )

    model, model_clean, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, model_clean, optimizer, train_dataloader, lr_scheduler
    )
    if config.data_from_model:
        generator = torch.Generator()
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(
            total=len(train_dataloader), disable=not accelerator.is_local_main_process
        )
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images = batch["images"] if not config.data_from_model else batch[0]
            device = clean_images.device

            if config.l_inf_noise > 0:
                target_images = (
                    clean_images + torch.rand_like(clean_images) * config.l_inf_noise
                )
            elif config.add_pertruebation:
                adversrial_images = attack.generate(clean_images.cpu().numpy())
                adversrial_images = torch.from_numpy(adversrial_images).to(device)
                target_images = adversrial_images
                if global_step == 0:
                    evaluate_diff_algo(
                        to_pil(adversrial_images),
                        to_pil(clean_images),
                        device,
                        accelerator,
                        global_step,
                        classifier,
                        labels=batch["labels"],
                    )
            else:
                target_images = clean_images

            target_images = normalize(target_images)

            generator = (
                generator.set_state(batch[1]) if config.data_from_model else None
            )  # generate the same noize as was inputted to the frozen model
            noise = torch.randn(target_images.shape, generator=generator).to(device)
            bs = target_images.shape[0]

            logging.info(f"num train timesteps: {config.training_steps}")
            timesteps = torch.randint(
                0,
                config.training_steps,
                (bs,),
                device=device,
            ).long()

            (
                noisy_images,
                sqrt_alpha_prod,
                sqrt_one_minus_alpha_prod,
            ) = noise_scheduler.add_noise(target_images, noise, timesteps)
            if config.train_clean_model:
                clean_images = normalize(clean_images)
                noisy_non_perturbed_images, _, _ = noise_scheduler.add_noise(
                    clean_images, noise, timesteps
                )
            with accelerator.accumulate(model):
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                reg_loss = None
                if config.regularize_using_clean_model:
                    with torch.no_grad():
                        model_clean_pred = model_clean(
                            noisy_images, timesteps, return_dict=False
                        )[0]
                    reg_loss = F.mse_loss(
                        noise_pred,
                        model_clean_pred,
                        reduce=not (config.hinge_loss_threshold > 0),
                    )

                    if config.hinge_loss_threshold > 0:
                        normalization = torch.clamp(
                            input=(sqrt_one_minus_alpha_prod / sqrt_alpha_prod) ** 2,
                            min=None,
                            max=50,
                        )
                        reg_loss = normalization * reg_loss
                        reg_loss = hinge_loss(
                            reg_loss,
                            config.hinge_loss_threshold,
                            1.0,
                        )
                        reg_loss = reg_loss.mean()

                    reg_loss = config.reg_lambda * reg_loss
                    loss += reg_loss
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if config.train_clean_model:
                with accelerator.accumulate(model_clean):
                    noise_pred = model_clean(
                        noisy_non_perturbed_images, timesteps, return_dict=False
                    )[0]
                    loss = F.mse_loss(noise_pred, noise)
                    accelerator.backward(loss)

                    accelerator.clip_grad_norm_(model_clean.parameters(), 1.0)
                    optimizer_clean.step()
                    lr_scheduler_clean.step()
                    optimizer_clean.zero_grad()

            progress_bar.update(1)
            logs = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "step": global_step,
            }
            if reg_loss is not None:
                logs["reg_loss"] = reg_loss.detach().item()
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            if config.train_clean_model:
                accelerator.log(
                    {"clean model loss": loss.detach().item()},
                    step=global_step,
                )
            global_step += 1

        if accelerator.is_main_process:
            pipeline_train = DDIMPipeline(
                unet=accelerator.unwrap_model(model), scheduler=noise_scheduler
            )
            pipeline_clean = DDIMPipeline(
                unet=accelerator.unwrap_model(model_clean), scheduler=noise_scheduler
            )

            if (
                epoch + 1
            ) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                images_predicted = evaluate(config, epoch, pipeline_train)
                if config.train_clean_model:
                    original_images_ = evaluate(
                        config, epoch, pipeline_clean, postfix="clean"
                    )

                evaluate_diff(
                    images_predicted,
                    original_images
                    if not config.train_clean_model
                    else original_images_,
                    device,
                    accelerator,
                    global_step,
                    classifier,
                    log_clean=config.train_clean_model,
                    labels=batch["labels"] if config.labels_from_data else None,
                )

            if (
                epoch + 1
            ) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                pipeline_train.save_pretrained(
                    os.path.join(config.output_dir, config.run_name)
                )


from accelerate import notebook_launcher

args = (
    config,
    model,
    noise_scheduler,
    optimizer,
    train_dataloader,
    lr_scheduler,
    model_clean
    if config.train_clean_model or config.regularize_using_clean_model
    else None,
    optimizer_clean if config.train_clean_model else None,
    lr_scheduler_clean if config.train_clean_model else None,
)


notebook_launcher(train_loop, args, num_processes=1)
