## Training configuration

# For convenience, create a `TrainingConfig` class containing the training hyperparameters (feel free to adjust them):

from dataclasses import dataclass
import dataclasses
import logging
import torch.nn.functional as F
import wandb

from models import DDIMPipelineGivenImage, DDPMPipelineGivenImage
from diffusers import DDIMInverseScheduler, DDIMPipeline, DDIMScheduler, DDPMScheduler
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
from diffusers.utils import numpy_to_pil
from accelerate.utils import ProjectConfiguration
from datasets import load_dataset
from torchvision import transforms


# from accelerate.logging import get_logger

logging.basicConfig(format="%(message)s", level=logging.INFO)
cuda = torch.device("cuda")


class ConvNetClassifier(nn.Module):
    def __init__(self):
        super(ConvNetClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


class MnistClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self._net = ConvNetClassifier()
        self._net.load_state_dict(torch.load("model.pth"))
        self._net.eval()
        self._net.to("cuda")
        self._transforms_tensor = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        self._transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

    def preprocess(self, x):
        if not isinstance(x, torch.Tensor):
            return torch.stack([self._transforms_tensor(image) for image in x]).to(cuda)

        # return self._transforms(x)
        return x

    def forward(self, x):
        pred = self._net(x)
        return pred


if __name__ == "__main__":
    classifier_ = MnistClassifier()


    @dataclass
    class TrainingConfig:
        train_batch_size = 16
        eval_batch_size = 16
        num_epochs = 50
        gradient_accumulation_steps = 1
        learning_rate = 1e-4
        lr_warmup_steps = 500
        save_image_epochs = 1
        save_model_epochs = 350
        mixed_precision = "no"  # `no` for float32, `fp16` for automatic mixed precision
        output_dir = "noisy-model-test"  # the model name locally and on the HF Hub
        seed = 0
        run_name = "model pretraining testing"
        logdir = "logs"
        num_steps_train = 1000
        num_steps_original = 1000

        def to_dict(self) -> dict:
            return dataclasses.asdict(self)


    model_id = "nabdan/mnist_20_epoch"

    pipeline_orig = DDIMPipelineGivenImage.from_pretrained(model_id).to(cuda)
    pipeline_orig.scheduler = DDIMScheduler.from_config(pipeline_orig.scheduler.config)

    pipeline_train = DDIMPipelineGivenImage.from_pretrained(model_id).to(cuda)
    pipeline_train.scheduler = DDIMScheduler.from_config(pipeline_train.scheduler.config)

    config = TrainingConfig()

    pipeline_train.scheduler.set_timesteps(config.num_steps_train)
    pipeline_orig.scheduler.set_timesteps(config.num_steps_original)

    model = pipeline_train.unet

    to_tensor = torchvision.transforms.ToTensor()
    # num_training_iterations = 10

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

    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    dataset_name = "mnist"
    dataset = load_dataset(dataset_name, split="train")


    def transform(examples):
        images = [preprocess(image.convert("L")) for image in examples["image"]]
        return {"images": images}


    dataset.set_transform(transform)
    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=config.train_batch_size, shuffle=True
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * config.num_epochs),
    )

    _, _, min_pixel_value, max_pixel_value = load_mnist()
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


    def save_images(images, epoch, postfix=None):
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


    def evaluate(config, pipeline, postfix=None):
        model.eval()
        images = pipeline(
            batch_size=config.eval_batch_size,
            generator=torch.manual_seed(config.seed),
            num_inference_steps=pipeline.scheduler.num_inference_steps,
        ).images

        # save_images(images, epoch, postfix=postfix)
        return images


    original_images = evaluate(
        config,
        pipeline_orig,
        postfix="orig",
    )


    def evaluate_diff(
        images_noisy, original_images, config, epoch, device, accelerator, global_step
    ):
        with torch.no_grad():
            l2_diff = l2_norm_diff(images_noisy, original_images, device=device)
            l_inf_diff = l_inf_norm_diff(images_noisy, original_images, device=device)
            logging.info(f"l2_diff: {l2_diff}")
            logging.info(f"l_inf_diff: {l_inf_diff}")
            logging.info(f"l2_diff mean: {l2_diff.mean().item()}")
            logging.info(f"l_inf_diff mean: {l_inf_diff.mean().item()}")

            original_scores = classifier_.forward(classifier_.preprocess(original_images))
            labels = original_scores.argmax(dim=1)

            ft_scores = classifier_.forward(classifier_.preprocess(images_noisy))
            ft_labels = ft_scores.argmax(dim=1)
            accuracy = (ft_labels == labels).float().mean()
            logging.info(f"accuracy: {accuracy.item()}")

            accelerator.log(
                {
                    "l2_diff": l2_diff.mean().item(),
                    "l_inf_diff": l_inf_diff.mean().item(),
                    "step": global_step,
                    "l_inf_diff_hist": l_inf_diff.flatten(),
                    "l2_diff_hist": l2_diff.flatten(),
                    "accuracy": accuracy.item(),
                },
                step=global_step,
            )

            accelerator.log(
                {
                    "predicted pertrubed images": [
                        wandb.Image(image) for image in images_noisy[:4]
                    ],
                },
                step=global_step,
            )


    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="wandb",
        project_dir=os.path.join(config.output_dir, "logs"),
        project_config=ProjectConfiguration(
            project_dir=os.path.join(config.logdir, config.run_name),
            automatic_checkpoint_naming=True,
        ),
    )
    accelerator.init_trackers(
        project_name="ddpo-pytorch-pretraining",
        config=config.to_dict(),
        init_kwargs={"wandb": {"name": config.run_name}},
    )
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers("train_example")

    accelerator.log(
        {
            "original images": [wandb.Image(image) for image in original_images[:4]],
        },
    )

    # model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)

    global_step = 0

    attack = FastGradientMethod(
        estimator=classifier, eps=0.3, eps_step=0.01, batch_size=config.train_batch_size
    )
    # img_prcoessor: ViTImageProcessor = pipe.image_processor


    # train_dataloader_iter = iter(train_dataloader)

    for epoch in range(config.num_epochs):
        progress_bar = tqdm(
            total=len(train_dataloader), disable=not accelerator.is_local_main_process
        )
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images = batch["images"].to(cuda)
            # for step in range(num_training_iterations):
            model.train()
            # clean_images_pil = pipeline_orig(
            #     batch_size=config.train_batch_size,
            #     num_inference_steps=pipeline_orig.scheduler.num_inference_steps,
            #     # image_=noise,
            # ).images
            # clean_images = torch.stack([to_tensor(image) for image in clean_images_pil]).to(
            #     cuda
            # )

            # try:
            #     clean_images = next(train_dataloader_iter)["images"].to(cuda)
            # except StopIteration:
            #     train_dataloader_iter = iter(train_dataloader)
            #     clean_images = next(train_dataloader_iter)["images"].to(cuda)

            # clean_images = img_prcoessor.preprocess(clean_images)
            # clean_images = classifier_.preprocess(clean_images)

            # adversrial_images = attack.generate(clean_images.cpu().numpy())

            adversrial_images = clean_images.cpu().numpy()

            # adversrial_images_pil = numpy_to_pil(adversrial_images.transpose(0, 2, 3, 1))

            # accelerator.log(
            #     {
            #         "original adversarial images": [
            #             wandb.Image(image) for image in adversrial_images_pil[:2]
            #         ],
            #     },
            #     step=global_step,
            # )

            adversrial_images = torch.from_numpy(adversrial_images).to(cuda)

            bs = clean_images.shape[0]

            timesteps = torch.randint(
                0,
                pipeline_train.scheduler.config.num_train_timesteps,
                (bs,),
                device=clean_images.device,
            ).long()

            noise = torch.randn(size=adversrial_images.shape).to(cuda)
            noisy_images = pipeline_train.scheduler.add_noise(
                adversrial_images, noise, timesteps
            )

            # noisy_images_pil = numpy_to_pil(
            #     noisy_images[:2, :].detach().cpu().numpy().transpose(0, 2, 3, 1)
            # )

            # accelerator.log(
            #     {
            #         "noisy images": [
            #             wandb.Image(image, caption=f"timestemp: {timestep}")
            #             for image, timestep in zip(noisy_images_pil, timesteps[:2])
            #         ],
            #     },
            #     step=global_step,
            # )

            with accelerator.accumulate(model):
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                logging.info(f"loss: {loss.item()}")
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # noise_pred_pil = numpy_to_pil(
            #     noise_pred[:2, :].detach().cpu().numpy().transpose(0, 2, 3, 1)
            # )

            # accelerator.log(
            #     {
            #         "noise prediction images": [
            #             wandb.Image(image) for image in noise_pred_pil
            #         ],
            #     },
            #     step=global_step,
            # )

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
            images_noisy = evaluate(config, pipeline_train)
            logging.info("model images")
            evaluate_diff(
                images_noisy,
                original_images,
                epoch,
                config,
                clean_images.device,
                accelerator,
                global_step,
            )

            if (
                epoch + 1
            ) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                pipeline.save_pretrained(config.output_dir)