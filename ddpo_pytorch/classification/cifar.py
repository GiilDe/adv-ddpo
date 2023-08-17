from typing import Any
import torch
import torchvision
from torchvision.models import efficientnet, EfficientNet_V2_S_Weights
from torch import functional as F
from torch import nn

from ddpo_pytorch.classification.classifier import Classifier
from transformers import pipeline
from transformers.pipelines.image_classification import ImageClassificationPipeline


class CifarClassifier(Classifier):
    def __init__(self) -> None:
        self.pipe: ImageClassificationPipeline = pipeline(
            "image-classification", model="tzhao3/vit-CIFAR10"
        )
        self.pipe.model.eval()

    def preprocess(self, pil_image) -> torch.Tensor:
        return self.pipe.preprocess(pil_image)["pixel_values"].squeeze(0)

    def predict(self, tensor_images) -> torch.Tensor:
        outputs = self.pipe.model(tensor_images)
        probs = outputs.logits.softmax(-1)
        return probs
