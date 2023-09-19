

## Installation
Requires Python 3.10 or newer.

```bash
git clone git@github.com:GiilDe/adv-ddpo.git
pip install -e .
```

## Usage
Run
```bash
accelerate launch scripts/train.py --config "config/cifar.py"
```
To run training on CIFAR. You can also run on MNIST by using `config/base.py` instead.
