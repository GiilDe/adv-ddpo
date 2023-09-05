
from data.cifar10 import Cifar10Dataset

def init_by_dataset(dataset_name, batch_size, root, shuffle=True, train=False, download=True):
    if dataset_name == "CIFAR10":
        return Cifar10Dataset(batch_size, root, shuffle=True, train=False, download=True)

    else:
        raise ValueError("Unrecognized dataset name.")
