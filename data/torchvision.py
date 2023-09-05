import torch
import torchvision
from abc import ABC, abstractmethod


class TorchvisionDataset(ABC):


  @abstractmethod  
  def __init__(self, batch_size, root, shuffle=True, train=False, download=True):
    pass

  @property
  @abstractmethod
  def dataloader(self):
    return self._dataloader
  
  @property
  @abstractmethod
  def dataset(self):
    return self._dataset
