 

class SharedNN:
    def __init__(self, models, optimizers):
        self.models = models
        self.optimizers = optimizers

        self.data = [] # will contain [output of encoder, output of decoder]
        self.remote_tensors = []

    def forward(self, encoder_index, x):
        data = []
        remote_tensors = []

        data.append(self.models[encoder_index](x))

        # self.models[-1] is the decoder
        remote_tensors.append(data[0].detach().move(self.models[-1].location).requires_grad_())

        data.append(self.models[-1](remote_tensors[-1]))

        self.data = data
        self.remote_tensors = remote_tensors

        return data[-1]

    def backward(self):
        # remote_tensors[0].location is on the decoder
        #grads = self.remote_tensors[0].grad.copy().move(self.data[0].location)
        g1 = self.remote_tensors[0]
        g2 = g1.grad
        print(type(g2))
        print(g2.shape)
        print(g1.copy().grad.get())
        g3 = g2.copy()
        g4 = g3.move(self.data[0].location)
        grads = g4

        self.data[0].backward(grads)

    def zero_grads(self):
        for opt in self.optimizers:
            opt.zero_grad()

    def step(self):
        for opt in self.optimizers:
            opt.step()


