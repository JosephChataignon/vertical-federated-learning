 
import sys
import utils

PYVERTICAL_LOCATION = utils.load_from_dotenv("PYVERTICAL_LOCATION")
sys.path.append(PYVERTICAL_LOCATION)

import torch
from torchvision import datasets, transforms
from torch import nn, optim
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

import syft as sy
import matplotlib.pyplot as plt

from src.psi.util import Client, Server
from src.utils import add_ids
from data_loader import VerticalDataLoader

hook = sy.TorchHook(torch)




# create dataset










torch.manual_seed(0)




