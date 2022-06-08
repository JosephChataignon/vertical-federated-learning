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

from autoencoder import Autoencoder

hook = sy.TorchHook(torch)


## Create dataset

# add_ids is a wrapper that adds ids to whatever torch dataset is given in argument
data = add_ids(MNIST)(".", download=True, transform=ToTensor())

# Batch data
dataloader = VerticalDataLoader(data, remove_data=False, keep_order=True, batch_size=128) 
# partition_dataset uses by default "remove_data=True, keep_order=False"
# Do not do this for now

# Plot the first 10 entries of the dataset image1 (top-left corner of images)
figure = plt.figure()
for index in range(1, 11):
    plt.subplot(6, 10, index)
    plt.axis('off')
    plt.imshow(dataloader.dataloader_image1.dataset.data[index].numpy().squeeze(), cmap='gray_r')
    print(dataloader.dataloader_labels.dataset[index][0], end=" ")
plt.show()




## Compute private set intersection (PSI)
# this section will turn useful in scenarios when not all actors have the same data points
client1_items = dataloader.dataloader_image1.dataset.get_ids()
client2_items = dataloader.dataloader_image2.dataset.get_ids()
client3_items = dataloader.dataloader_image3.dataset.get_ids()
client4_items = dataloader.dataloader_image4.dataset.get_ids()
server_items  = dataloader.dataloader_labels.dataset.get_ids()

client1 = Client(client1_items)
client2 = Client(client2_items)
client3 = Client(client3_items)
client4 = Client(client4_items)
server  = Server(server_items)

#setup, response = server.process_request(client.request, len(client_items))
#intersection = client.compute_intersection(setup, response)

# Order data
#dataloader.drop_non_intersecting(intersection)
dataloader.sort_by_ids()



torch.manual_seed(0)





## Build and train autoencoders

autoencoder_1 = Autoencoder(dataloader.dataloader_image1)
autoencoder_2 = Autoencoder(dataloader.dataloader_image2)
autoencoder_3 = Autoencoder(dataloader.dataloader_image3)
autoencoder_4 = Autoencoder(dataloader.dataloader_image4)

autoencoder_1.train_model()
autoencoder_2.train_model()
autoencoder_3.train_model()
autoencoder_4.train_model()



# DataLoaders that give the 4 quarters of images
# keep image1 and pass others through autoencoderX.encoder
# make a new dataloader out of it
# new network using this as entry to rebuild image1



