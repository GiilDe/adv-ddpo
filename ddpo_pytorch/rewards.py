from PIL import Image
import io
import numpy as np
import torch
import torchvision
from torch import functional as F
from torch import nn

from ddpo_pytorch.classification.classifier import Classifier

to_tensor = torchvision.transforms.ToTensor()

C = 0.03


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
    original_images_ = torch.stack([to_tensor(image) for image in original_images]).to("cuda")
    ft_images_ = torch.stack([to_tensor(image) for image in ft_images]).to("cuda")

    image_size = 1
    for dim_ in ft_images_.shape[1:]:
        image_size *= dim_

    images_diff = (
        torch.linalg.vector_norm(ft_images_ - original_images_, ord=2, dim=(1, 2, 3))
        / image_size
    )
    return images_diff


def l_inf_norm_diff(ft_images, original_images):
    original_images_ = torch.stack([to_tensor(image) for image in original_images]).to("cuda")
    ft_images_ = torch.stack([to_tensor(image) for image in ft_images]).to("cuda")

    images_diff = torch.linalg.vector_norm(
        ft_images_ - original_images_, ord=float("inf"), dim=(1, 2, 3)
    )
    return images_diff


def gen_reward_fn(l_for_penalty, config, classifier: Classifier):
    def _fn(ft_images, original_images, metadata):
        assert l_for_penalty in ["l_inf", "l2"]
        with torch.no_grad():
            original_images_ = torch.stack(
                [classifier.preprocess(image) for image in original_images]
            ).to("cuda")
            ft_images_ = torch.stack(
                [classifier.preprocess(image) for image in ft_images]
            ).to("cuda")
            original_scores = classifier.predict(original_images_)
            labels = original_scores.argmax(dim=1)

            ft_scores = classifier.predict(ft_images_)
            ft_labels_scores = torch.gather(ft_scores, 1, labels.unsqueeze(1))
            ft_labels_scores = ft_labels_scores.reshape(
                len(ft_images)
            )  # remove useless dimensions

            mask = torch.ones_like(ft_scores).scatter_(1, labels.unsqueeze(1), 0.0)
            max_scores = None
            if config.reward_type == "hinge-reward":
                max_scores = (
                    ft_scores[mask.bool()]
                    .view(ft_scores.shape[0], ft_scores.shape[1] - 1)
                    .max(dim=1)[0]
                )

            ft_labels = ft_scores.argmax(dim=1)
            accuracy = (ft_labels == labels).float().mean()

            images_diff_l2 = l2_norm_diff(ft_images, original_images)
            images_diff_l_inf = l_inf_norm_diff(ft_images, original_images)
            images_distance = (
                images_diff_l_inf if l_for_penalty == "l_inf" else images_diff_l2
            )
            images_penalty = (
                hinge_loss(
                    images_distance,
                    config.images_diff_threshold,
                    config.images_diff_weight,
                )
                if config.images_diff_weight > 0.0
                else 0.0
            )

            if config.reward_type == "hinge-reward":
                reward = torch.minimum(
                    torch.zeros_like(max_scores), max_scores - ft_labels_scores - C
                )
            elif config.reward_type == "log-reward":
                reward = torch.log(1 - ft_labels_scores)
            else:
                reward = 1 - ft_labels_scores
            assert reward.isnan().sum() == 0
            return reward - images_penalty, {
                "ft_labels_scores": ft_labels_scores,
                "images_diff_l2": images_diff_l2,
                "images_diff_l_inf": images_diff_l_inf,
                "accuracy": accuracy,
            }

    return _fn


def untargeted_l2_img_diff(config, classifier):
    return gen_reward_fn("l2", config, classifier)


def untargeted_l_inf_img_diff(config, classifier):
    return gen_reward_fn("l_inf", config, classifier)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
