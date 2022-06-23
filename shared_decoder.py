 
import sys, copy
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
from class_split_data_loader import ClassSplitDataLoader
from shared_NN import SharedNN

hook = sy.TorchHook(torch)




## Create dataset

n_encoders = 3 # number of encoders we will train

# add_ids is a wrapper that adds ids to whatever torch dataset is given in argument
data = add_ids(MNIST)(".", download=True, transform=ToTensor())

# Create dataloader(s)
dataloaders = []
for k in range(n_encoders):
    dataloader = ClassSplitDataLoader(data, class_to_keep=k, remove_data=False, keep_order=True, batch_size=128) 
    dataloaders.append(dataloader)
    # partition_dataset uses by default "remove_data=True, keep_order=False"
    # Do not do this for now




## Build Neural Network

epochs = 10

torch.manual_seed(0)

input_size = 784
hidden_sizes = [128, 640]
encoded_size = 10

encoder = nn.Sequential(
        nn.Linear(input_size, hidden_sizes[0]),
        nn.ReLU(),
        nn.Linear(hidden_sizes[0], hidden_sizes[1]),
        nn.ReLU(),
        nn.Linear(hidden_sizes[1], encoded_size),
        nn.ReLU(),
    )
decoder = nn.Sequential(
        nn.Linear(encoded_size, hidden_sizes[1]),
        nn.ReLU(),
        nn.Linear(hidden_sizes[1], hidden_sizes[0]),
        nn.ReLU(),
        nn.Linear(hidden_sizes[0], input_size),
        nn.LogSoftmax(dim=1),
    )
buffer_layer = nn.Identity(input_size, unused_argument1=0.1)
encoders = [copy.deepcopy(encoder) for k in range(n_encoders)]
buffer_layers = [copy.deepcopy(buffer_layer) for k in range(n_encoders)]

# Create optimisers for each segment and link to them
models_to_learn = encoders + [decoder]
optimizers = [optim.SGD(model.parameters(), lr=0.03,) for model in models_to_learn]

# create some workers
encoder_workers = [sy.VirtualWorker(hook, id=f"encoder_{k}") for k in range(n_encoders)]
decoder_worker = sy.VirtualWorker(hook, id="decoder")

# Send Model Segments to model locations
for model, location in zip(encoders, encoder_workers):
    print(model)
    model.send(location)
    print(model.location)
decoder.send(decoder_worker)
for model, location in zip(buffer_layers, encoder_workers):
    print(model)
    model.send(location)
    print(model.location)

print('DEBUG:: ')
print(buffer_layers[0].location)

# Create the SharedNN
sharedNN = SharedNN(encoders, decoder, buffer_layers, optimizers)


## Learning

for i in range(epochs):
    running_loss = 0
    correct_preds = 0
    total_preds = 0
    
    for k in range(n_encoders):
        # for now, train the encoders one after another
        dataloader = dataloaders[k]
        
        for ((data, ids),) in dataloader:
            # Train a model
            data = data.view(data.shape[0], -1)
            data_for_comparison = copy.deepcopy(data)
            # we need a copy of the data to compare to the output of the decoder
            data = data.send(encoders[k].location)
            data_for_comparison = data_for_comparison.send(encoders[k].location)

            #1) Zero our grads
            sharedNN.zero_grads()
            
            #2) Make a prediction and move it to the encoder
            pred = sharedNN.forward(k, data)
            #pred = pred.move(models[k].location)
            
            #3) Figure out how much we missed by
            criterion = nn.MSELoss()
            loss = criterion(pred, data)
            
            #4) Backprop the loss on the end layer
            #loss = loss.move(models[-1].location)
            #print(f'loss grad: {loss.copy().get().grad}')
            loss.backward()
            
            #5) Feed Gradients backward through the nework
            sharedNN.backward()
            
            #6) Change the weights
            sharedNN.step()

    # Collect statistics
    running_loss += loss.get()
    #correct_preds += pred.max(1)[1].eq(labels).sum().get().item()
    total_preds += pred.get().size(0)

    print(f"Epoch {i} - Training loss: {running_loss/len(dataloader):.3f} - Accuracy: {100*correct_preds/total_preds:.3f}")
































