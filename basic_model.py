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

from data_loader import VerticalDataLoader
from src.psi.util import Client, Server
from src.utils import add_ids


hook = sy.TorchHook(torch)


## Create dataset

# add_ids is a wrapper that adds ids to whatever torch dataset is given in argument
data = add_ids(MNIST)(".", download=True, transform=ToTensor())

# Batch data
dataloader = VerticalDataLoader(data, batch_size=128) 
# partition_dataset uses by default "remove_data=True, keep_order=False"

# Plot the first 10 entries of the dataset image1 (top-left corner of images)
figure = plt.figure()
for index in range(1, 11):
    plt.subplot(6, 10, index)
    plt.axis('off')
    plt.imshow(dataloader.dataloader_image1.dataset.data[index].numpy().squeeze(), cmap='gray_r')
plt.show()


