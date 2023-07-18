from PIL import Image
import io
import numpy as np
import torch

from models import MnistClassifier
import torchvision
from torch import functional as F


def targeted_mnist_classifier():
    target = 3
    images_diff_weight = 0.1
    mnist_classifier = MnistClassifier()
    mnist_classifier.load_state_dict(torch.load("model.pth"))
    mnist_classifier.eval()

    to_tensor = torchvision.transforms.ToTensor()

    mnist_pil_to_tensor = torchvision.transforms.Compose(
        [
            to_tensor,
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    def _fn(ft_images, original_images, metadata):
        with torch.no_grad():
            ft_images_ = [mnist_pil_to_tensor(image) for image in ft_images]
            ft_images_ = torch.stack(ft_images_)
            pred = mnist_classifier(ft_images_)
            target_scores = pred[:, target]

            ft_images_ = torch.stack([to_tensor(image) for image in ft_images])
            original_images_ = torch.stack([to_tensor(image) for image in original_images])
            images_diff = torch.norm(ft_images_ - original_images_)
            return target_scores - images_diff_weight*images_diff, {}

    return _fn