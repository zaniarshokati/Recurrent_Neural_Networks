import numpy as np
from Layers.Base import BaseLayer


class ReLU(BaseLayer):
    def __init__(self, testing_phase=False):
        super().__init__(testing_phase)
        self.input_tensor = None

    def forward(self, input_tensor):

        self.input_tensor = input_tensor
        self.input_tensor = np.maximum(0, self.input_tensor)
        return self.input_tensor

    def backward(self, error_tensor):
        input_mask = self.input_tensor > 0
        temp = error_tensor * input_mask
        return temp