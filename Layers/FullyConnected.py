import numpy as np
from Layers.Base import BaseLayer


class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size, testing_phase=False):
        super().__init__(testing_phase)
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.uniform(0, 1, size=(input_size + 1, output_size))

    def initialize(self, weights_initializer, bias_initializer):
        #examples for "weights_initializer" would be "Xavier"
        weights_shape = (self.input_size, self.output_size)
        bias_shape = (1, self.output_size)
        weights = weights_initializer.initialize(weights_shape, self.input_size, self.output_size)
        bias = bias_initializer.initialize(bias_shape, 1, self.output_size)
        self.weights = np.vstack((bias, weights))

    def forward(self, input_tensor):
        temp = np.ones((np.shape(input_tensor)[0], 1))
        self.input_tensor = np.append(temp, input_tensor, axis=1)
        output = np.dot(self.input_tensor, self.weights)
        return output

    def backward(self, error_tensor):
        self.gradient_weights = np.dot(self.input_tensor.T, error_tensor)

        if self.optimizer is not None:
            self.weights = self._optimizer.calculate_update(self._weights, self._gradient_weights)

        error_tensor = np.dot(error_tensor, self.weights[1:, :].T)  # gradient_wrt_input

        return error_tensor

