
import torch
from torchvision import datasets, transforms
from torch import nn, optim
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor




class Autoencoder(nn.Module):
    def __init__(self, dataloader, epochs=100, batchSize=128):
        super(Autoencoder, self).__init__()
        self.dataloader = dataloader # should be dataloader.dataloader_imageX
        self.epochs = epochs
        self.batchSize = batchSize
        
        input_size = 196 # images are 1/4 of MNIST, so 14*14=196
        hidden_sizes_encoder = [50, 20]
        hidden_sizes_decoder = [20, 50]
        output_size = 11

        self.encoder = nn.Sequential(
                nn.Linear(input_size, hidden_sizes_encoder[0]),
                nn.ReLU(),
                nn.Linear(hidden_sizes_encoder[0], hidden_sizes_encoder[1]),
                nn.ReLU(),
                nn.Linear(hidden_sizes_encoder[1], output_size)
            )
        self.decoder = nn.Sequential(
                nn.Linear(output_size, hidden_sizes_decoder[0]),
                nn.ReLU(),
                nn.Linear(hidden_sizes_decoder[0], hidden_sizes_decoder[1]),
                nn.ReLU(),
                nn.Linear(hidden_sizes_decoder[1], input_size)
            )

        # Create optimiser
        self.optimizer = optim.SGD(self.parameters(), lr=0.03,)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def train_model(self):
        for epoch in range(self.epochs):
            for data in self.dataloader:
                image, ids = data
                image = image.view(image.shape[0],-1)
                # predictions and loss
                output = self(image)
                loss = self.criterion(output,image)
                # backprop
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            print(f'epoch {epoch}/{self.epochs}, loss: {loss.data}')


 
