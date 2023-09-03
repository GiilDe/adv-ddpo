import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

from ddpo_pytorch.classification.classifier import Classifier


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
        return F.softmax(x, dim=1)


class MnistClassifier(Classifier):
    def __init__(self):
        self._net = ConvNetClassifier()
        self._net.load_state_dict(torch.load("model.pth"))
        self._net.eval()
        self.device = torch.device("cuda:0")
        self._net.to(self.device)
        self._transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

    def preprocess(self, x):
        return self._transforms(x)

    def predict(self, x):
        pred = self._net(x)
        return pred
