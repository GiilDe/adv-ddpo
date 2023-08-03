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


def hinge_loss(images_distance, images_diff_threshold, images_diff_weight):
    hinge_distance = (
        torch.maximum(
            torch.zeros_like(images_distance),
            images_distance - images_diff_threshold,
        )
        if (isinstance(images_diff_threshold, float) and images_diff_threshold > 0.0)
        else images_distance
    )
    return hinge_distance * images_diff_weight

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


def gen_reward_fn(l_for_penalty, config):
    def _fn(ft_images, original_images, metadata):
        assert l_for_penalty in ["l_inf", "l2"]
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

            images_diff_l2 = l2_norm_diff(ft_images, original_images)
            images_diff_l_inf = l_inf_norm_diff(ft_images, original_images)
            images_distance = (
                images_diff_l_inf if l_for_penalty == "l_inf" else images_diff_l2
            )
            images_penalty = hinge_loss(images_distance, config.images_diff_threshold, config.images_diff_weight) if config.images_diff_weight > 0.0 else 0.0
            return torch.log(1 - ft_labels_scores) - images_penalty, {
                "ft_labels_scores": ft_labels_scores,
                "images_diff_l2": images_diff_l2,
                "images_diff_l_inf": images_diff_l_inf,
                "accuracy": accuracy,
            }

    return _fn


def untargeted_l2_img_diff(config):
    return gen_reward_fn("l2", config)


def untargeted_l_inf_img_diff(config):
    return gen_reward_fn("l_inf", config)
