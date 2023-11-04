import numpy as np
from Layers.Base import BaseLayer


class Flatten(BaseLayer):
    def __init__(self, testing_phase=False):
        super().__init__(testing_phase)
        self.input_shape = None

    def forward(self, input_tensor):
        self.input_shape = np.shape(input_tensor)
        batch_size = self.input_shape[0]
        matrix_size = np.size(input_tensor)
        col_size = int(matrix_size / batch_size)
        output = np.reshape(input_tensor, (batch_size, col_size))
        return output

    def backward(self, error_tensor):
        output = np.reshape(error_tensor, self.input_shape)
        return output