import copy

class SharedNN:
    def __init__(self, encoders, decoder, buffer_layers, optimizers):
        self.encoders = encoders
        self.decoder = decoder
        self.buffer_layers = buffer_layers
        self.optimizers = optimizers

        self.data = [] # will contain [output of encoder, output of decoder]
        self.remote_tensors = []

    def forward(self, encoder_index, x):
        data = []
        remote_tensors = []

        data.append(self.encoders[encoder_index](x))
        remote_tensors.append(data[0].detach().move(self.decoder.location).requires_grad_())
        
        data.append(self.decoder(remote_tensors[-1]))
        remote_tensors.append(data[1].detach().move(self.encoders[encoder_index].location).requires_grad_())

        data.append(self.buffer_layers[encoder_index](remote_tensors[-1]))

        self.data = data
        self.remote_tensors = remote_tensors

        return data[-1]

    def backward(self):
        
        grads = self.remote_tensors[1].grad.copy().move(self.data[1].location)
        self.data[1].backward(grads)
        print(self.data[1].copy().get().grad)
        
        grads = self.remote_tensors[0].grad.copy().move(self.data[0].location)
        self.data[0].backward(grads)

    def zero_grads(self):
        for opt in self.optimizers:
            opt.zero_grad()

    def step(self):
        for opt in self.optimizers:
            opt.step()


