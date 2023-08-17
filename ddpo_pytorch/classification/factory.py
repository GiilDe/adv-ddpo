from ddpo_pytorch.classification.cifar import pipe
from ddpo_pytorch.classification.mnist import MnistClassifier


def init_by_dataset(dataset_name):
    if dataset_name == "MNIST":
      return MnistClassifier()
    
    if dataset_name == "CIFAR10":
      return pipe
    
    else:
      raise ValueError("Unrecognized dataset name.")