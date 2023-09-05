
from data.torchvision import TorchvisionDataset
import torch
import torchvision
import torchvision.transforms as transforms


class Cifar10Dataset(TorchvisionDataset):
  
  def __init__(self, batch_size, root, shuffle=False, train=False, download=True):
    self._transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    self._dataset = torchvision.datasets.CIFAR10(root=root, train=train, download=download, transform=self._transforms)
    self._dataloader = torch.utils.data.DataLoader(self._dataset, batch_size=batch_size, shuffle=shuffle)

  @property
  def dataloader(self):
    return self._dataloader
  
  @property
  def dataset(self):
    return self._dataset