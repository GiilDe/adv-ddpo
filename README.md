

## Installation
Requires Python 3.10 or newer.

```bash
git clone git@github.com:GiilDe/adv-ddpo.git
pip install -e .
```

## Usage
First run the inversion
```bash
python scripts/invert.py --config config/cifar_inversion.py
```
Then, run RL training
Run
```bash
accelerate launch scripts/train.py --config "config/cifar.py"
```
To run training on CIFAR. You can also run on MNIST by using `config/base.py` instead.

If you wish to experiment with pretraining, i.e learning to attack using diffusion loss, run:
```bash
python pretraining.py
``` 
After saving a pretrained model you can train it using RL by specifying its path in your config file.

The training uses wandb for logging. To see the logs, you need to click the link in the output of the training script.