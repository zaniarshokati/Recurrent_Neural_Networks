import copy

class NeuralNetwork:
    def __init__(self, optimizer, weights_initializer, bias_initializer):
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None
        self.optimizer = optimizer
        self.label_tensor = None

        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer

        self.phase = False

    def forward(self):
        input_tensor, label_tensor = self.data_layer.next()
        self.label_tensor = label_tensor
        extra_loss = 0
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
            if self.optimizer.regularizer is not None:
                extra_loss += self.optimizer.regularizer.norm(layer.weights)
        loss = self.loss_layer.forward(input_tensor, label_tensor)
        loss_tilda = loss + extra_loss
        return loss_tilda

    def backward(self):
        error_tensor = self.loss_layer.backward(self.label_tensor)
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)

    def append_trainable_layer(self, layer):
        #make the deep copy and set it for the new layer that we want to append
        layer.optimizer = copy.deepcopy(self.optimizer)
        layer.initialize(self.weights_initializer, self.bias_initializer)
        #append this layer to the list of the layers which holds the architecture
        self.layers.append(layer)

    def train(self, iterations=100):
        self.phase = False
        for it_ in range(iterations):
            loss_ = self.forward()
            self.loss.append(loss_)
            self.backward()

    def test(self, input_tensor):
        self.phase = True
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        return input_tensor

    @property
    def phase(self):
        return self.__phase

    @phase.setter
    def phase(self, phase):
        self.__phase = phase
        for layer in self.layers:
            layer.testing_phase = phase











