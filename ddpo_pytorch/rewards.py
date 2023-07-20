from PIL import Image
import io
import numpy as np
import torch

from models import MnistClassifier
import torchvision
from torch import functional as F

mnist_classifier = MnistClassifier()
to_tensor = torchvision.transforms.ToTensor()
mnist_classifier.load_state_dict(torch.load("model.pth"))
mnist_classifier.eval()
mnist_pil_to_tensor = torchvision.transforms.Compose(
    [
        to_tensor,
        torchvision.transforms.Normalize((0.1307,), (0.3081,)),
    ]
)


def gen_reward_fn(norm, config):
    def _fn(ft_images, original_images, metadata):
        with torch.no_grad():
            original_images_ = torch.stack(
                [mnist_pil_to_tensor(image) for image in original_images]
            )
            ft_images_ = torch.stack(
                [mnist_pil_to_tensor(image) for image in ft_images]
            )

            original_scores = mnist_classifier(original_images_)
            labels = original_scores.argmax(dim=1)

            ft_scores = mnist_classifier(ft_images_)
            ft_labels_scores = torch.gather(ft_scores, 1, labels.unsqueeze(1))
            ft_labels_scores = ft_labels_scores.reshape(
                len(ft_images)
            )  # remove useless dimensions

            ft_labels = ft_scores.argmax(dim=1)
            accuracy = (ft_labels == labels).float().mean()

            images_diff = norm(ft_images, original_images)
            images_penalty = (
                max(0, images_diff - config.images_diff_threshold)
                * config.images_diff_weight
                if config.images_diff_threshold != 0
                else images_diff * config.images_diff_weight
            )
            return (1 - ft_labels_scores) - images_penalty, {
                "ft_labels_scores": ft_labels_scores,
                "images_diff": images_diff,
                "accuracy": accuracy,
            }

    return _fn


def l2_norm_diff(ft_images, original_images):
    original_images_ = torch.stack([to_tensor(image) for image in original_images])
    ft_images_ = torch.stack([to_tensor(image) for image in ft_images])

    image_size = 1
    for dim_ in ft_images_.shape[1:]:
        image_size *= dim_

    images_diff = (
        torch.linalg.vector_norm(ft_images_ - original_images_, ord=2, dim=(1, 2, 3))
        / image_size
    )
    return images_diff


def l_inf_norm_diff(ft_images, original_images):
    original_images_ = torch.stack([to_tensor(image) for image in original_images])
    ft_images_ = torch.stack([to_tensor(image) for image in ft_images])

    images_diff = torch.linalg.vector_norm(
        ft_images_ - original_images_, ord=float("inf"), dim=(1, 2, 3)
    )
    return images_diff


def untargeted_l2_img_diff(config):
    return gen_reward_fn(l2_norm_diff, config)


def untargeted_l_inf_img_diff(config):
    return gen_reward_fn(l_inf_norm_diff, config)
