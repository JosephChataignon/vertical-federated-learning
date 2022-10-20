import copy

import torch
from torchvision import datasets, transforms
from torch import nn, optim
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor




class Autoencoder(nn.Module):
    def __init__(self, settings):
        super(Autoencoder, self).__init__()
        self.settings = settings
        self.n_encoders = settings['n_encoders']
        self.dataloaders = settings['dataloaders']
        self.dataloaders_test = settings['dataloaders_test']
        self.criterion = settings['criterion']
        
        input_size = self.settings['input_size']
        encoded_size = self.settings['encoded_size']
        hidden_sizes_encoder = self.settings['hidden_sizes_encoder']
        hidden_sizes_decoder = self.settings['hidden_sizes_decoder']

        encoder = nn.Sequential(
                nn.Linear(input_size, hidden_sizes_encoder[0]),
                nn.ReLU(),
                nn.Linear(hidden_sizes_encoder[0], hidden_sizes_encoder[1]),
                nn.ReLU(),
                nn.Linear(hidden_sizes_encoder[1], encoded_size)
            )
        decoder = nn.Sequential(
                nn.Linear(encoded_size, hidden_sizes_decoder[0]),
                nn.ReLU(),
                nn.Linear(hidden_sizes_decoder[0], hidden_sizes_decoder[1]),
                nn.ReLU(),
                nn.Linear(hidden_sizes_decoder[1], input_size)
            )
        
        # Create models
        self.models = [copy.deepcopy(encoder) for k in range(self.n_encoders)] + [decoder]

        # Create optimiser
        self.optimizers = [optim.Adam(model.parameters(), lr=1e-3,) for model in self.models]

    def forward(self, encoder_index, input_vector):
        encoder_output = self.models[encoder_index](input_vector)
        encoded_vector = encoder_output.clone().detach()
        encoded_vector = encoded_vector.requires_grad_()
        decoder = self.models[-1]
        return encoder_output, encoded_vector, decoder(encoded_vector)
    
    
    def backward(self, encoder_output, encoded_vector):
        grads = encoded_vector.grad.clone().detach()
        encoder_output.backward(grads)


    def iter_training_one_encoder(self, encoder_index):
        
        running_loss = 0
        dataloader = self.dataloaders[encoder_index]

        for ((data, ids),) in dataloader:
            # Train a model
            data = data.view(data.shape[0], -1)
            
            #1) Zero our grads
            self.optimizers[encoder_index].zero_grad()
            
            #2) Make a prediction and move it to the encoder
            encoder_output, encoded_vector, pred = self.forward(encoder_index, data)
            
            #3) Figure out how much we missed by
            loss = self.criterion(pred, data)
            
            #4) Backprop the loss on the end layer
            loss.backward()
            
            #5) Feed Gradients backward through the network
            self.backward(encoder_output, encoded_vector)
            
            #6) Change the weights
            self.optimizers[encoder_index].step()

            # Collect statistics
            running_loss += loss.item()
        
        running_test_MSE = 0
        # Evaluate performance on test data
        dataloader_test = self.dataloaders_test[encoder_index]
        for ((data, ids),) in dataloader_test:
            data = data.view(data.shape[0], -1)
            pred = self.models[-1](self.models[encoder_index](data))
            #accuracy for an autoencoder is the distance between data and pred
            running_test_MSE += nn.MSELoss()(pred,data).item()
        
        return running_loss/len(dataloader), running_test_MSE/len(dataloader_test)
