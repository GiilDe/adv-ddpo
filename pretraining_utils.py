## Training configuration

# For convenience, create a `TrainingConfig` class containing the training hyperparameters (feel free to adjust them):

import logging
import torch.nn.functional as F
import wandb

import torch

import os
import torchvision

import torch
from ddpo_pytorch.rewards import l2_norm_diff, l_inf_norm_diff
import torch.nn as nn
from torchvision import transforms


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

        return self._transforms(x)

    def forward(self, x):
        pred = self._net(x)
        return pred


def evaluate_diff(
    images_predicted,
    original_images,
    device,
    accelerator,
    global_step,
    classifier,
    log_clean=False,
    labels=None,
    predicted_text="predicted pertrubed images",
):
    with torch.no_grad():
        l2_diff = l2_norm_diff(images_predicted, original_images, device=device)
        l_inf_diff = l_inf_norm_diff(images_predicted, original_images, device=device)
        logging.info(f"l2_diff: {l2_diff}")
        logging.info(f"l_inf_diff: {l_inf_diff}")
        logging.info(f"l2_diff mean: {l2_diff.mean().item()}")
        logging.info(f"l_inf_diff mean: {l_inf_diff.mean().item()}")

        if labels is not None:
            original_scores = classifier.forward(classifier.preprocess(original_images))
            labels = original_scores.argmax(dim=1)

        ft_scores = classifier.forward(classifier.preprocess(images_predicted))
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
                predicted_text: [wandb.Image(image) for image in images_predicted],
            },
            step=global_step,
        )

        if log_clean:
            accelerator.log(
                {
                    "original images": [
                        wandb.Image(image) for image in original_images
                    ],
                },
                step=global_step,
            )


def evaluate_diff_algo(
    images_predicted,
    original_images,
    device,
    accelerator,
    global_step,
    classifier,
    labels,
):
    with torch.no_grad():
        l2_diff = l2_norm_diff(images_predicted, original_images, device=device)
        l_inf_diff = l_inf_norm_diff(images_predicted, original_images, device=device)
        logging.info(f"l2_diff_algo: {l2_diff}")
        logging.info(f"l_inf_diff_algo: {l_inf_diff}")
        logging.info(f"l2_diff mean_algo: {l2_diff.mean().item()}")
        logging.info(f"l_inf_diff mean_algo: {l_inf_diff.mean().item()}")

        ft_scores = classifier.forward(classifier.preprocess(images_predicted))
        ft_labels = ft_scores.argmax(dim=1)
        accuracy = (ft_labels == labels).float().mean()
        logging.info(f"accuracy: {accuracy.item()}")

        accelerator.log(
            {
                "l2_diff_algo": l2_diff.mean().item(),
                "l_inf_diff_algo": l_inf_diff.mean().item(),
                "step": global_step,
                "l_inf_diff_hist_algo": l_inf_diff.flatten(),
                "l2_diff_hist_algo": l2_diff.flatten(),
                "accuracy_algo": accuracy.item(),
            },
            step=global_step,
        )

        accelerator.log(
            {
                "algo pertrubed images": [
                    wandb.Image(image) for image in images_predicted
                ],
            },
            step=global_step,
        )

        accelerator.log(
            {
                "clean images": [wandb.Image(image) for image in original_images],
            },
            step=global_step,
        )


to_tensor = transforms.ToTensor()


class PipelineIterator:
    def __init__(self, pipeline, batch_size, length):
        self._pipeline = pipeline
        self._batch_size = batch_size
        self.generator = torch.Generator()
        self.length = length
        self.counter = 0

    def __len__(self):
        return self.length

    def __next__(self):
        if self.counter >= len(self):
            self.counter = 0  # reset counter
            raise StopIteration
        generator_state = self.generator.get_state()
        images = self._pipeline(
            batch_size=self._batch_size,
            generator=self.generator,
            num_inference_steps=self._pipeline.scheduler.num_inference_steps,
        ).images
        self.counter += 1
        return (
            torch.stack([to_tensor(image) for image in images]).to("cuda"),
            generator_state,
        )

    def __iter__(self):
        return self
