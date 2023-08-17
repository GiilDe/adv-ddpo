import torch
import torchvision
from torchvision.models import efficientnet, EfficientNet_V2_S_Weights
from torch import functional as F
from torch import nn

from ddpo_pytorch.classification.classifier import Classifier

class CifarClassifier(Classifier):
    
    def __init__(self):
        self._weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
        self._net = torchvision.models.efficientnet_v2_s(weights=self._weights)
        self._transforms = self._weights.transforms()
        self._net.eval()

    def preprocess(self, x):
        return self._transforms(x)
    
    
    def predict(self, x):
        scores = self._net(x)
        pred = scores.squeeze(0).softmax(0)
        return pred
