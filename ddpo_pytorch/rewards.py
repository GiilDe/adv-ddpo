from PIL import Image
import io
import numpy as np
import torch

from models import MnistClassifier
import torchvision
from torch import functional as F

images_diff_weight = 0.3
mnist_classifier = MnistClassifier()
to_tensor = torchvision.transforms.ToTensor()
mnist_classifier.load_state_dict(torch.load("model.pth"))
mnist_classifier.eval()

def targeted_mnist_classifier():
    target = 3

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

            #ft_images_ = torch.stack([to_tensor(image) for image in ft_images])
            #original_images_ = torch.stack([to_tensor(image) for image in original_images])
            #images_diff = torch.norm(ft_images_ - original_images_)
            #return target_scores - images_diff_weight*images_diff, {}
            return target_scores, {}

    return _fn


def untargeted_mnist_classifier():

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
            
            original_images_ = [mnist_pil_to_tensor(image) for image in original_images]
            original_images_ = torch.stack(original_images_)

            original_scores = mnist_classifier(original_images_)
            
            labels = original_scores.argmax(dim=1)
            
            ft_scores = mnist_classifier(ft_images_)

            ft_labels_scores = torch.gather(ft_scores, 1, labels.unsqueeze(1))

            ft_images_ = torch.stack([to_tensor(image) for image in ft_images])
            original_images_ = torch.stack([to_tensor(image) for image in original_images])

            image_size = 1
            for dim_ in ft_images_.shape[1:]:
                image_size *= dim_
    
            images_diff = torch.linalg.vector_norm(ft_images_ - original_images_, ord=2, dim=(1,2,3))/image_size
            ft_labels_scores = ft_labels_scores.reshape(len(ft_images)) # remove useless dimensions
            return (1 - ft_labels_scores) - images_diff_weight*images_diff, {}


    return _fn