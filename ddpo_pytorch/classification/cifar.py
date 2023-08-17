import torch
import torchvision
from torchvision.models import efficientnet, EfficientNet_V2_S_Weights
from torch import functional as F
from torch import nn

from ddpo_pytorch.classification.classifier import Classifier
from transformers import pipeline

pipe = pipeline("image-classification", model="tzhao3/vit-CIFAR10")
