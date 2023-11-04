from Layers.Base import BaseLayer
import numpy as np


class Sigmoid(BaseLayer):
    def __init__(self):
        super().__init__()
        self.activations = None

    def forward(self, input_tensor):
        self.activations = 1 / (1 + np.exp(-1 * input_tensor))
        return self.activations

    def backward(self, error_tensor):
        return (self.activations * error_tensor) * (1 - self.activations)