from Layers.Base import BaseLayer
import numpy as np


class Dropout(BaseLayer):
    def __init__(self, probability=1, testing_phase=False):
        super().__init__(testing_phase)
        self.probability = probability
        self.cache_coef = None

    def forward(self, input_tensor):
        if self.testing_phase:
            return input_tensor
        shape_ = np.shape(input_tensor)
        rands = np.random.uniform(0, 1, shape_)
        mask = rands < self.probability
        self.cache_coef = mask / self.probability
        input_tensor = input_tensor * self.cache_coef
        return input_tensor

    def backward(self, error_tensor):
        return error_tensor * self.cache_coef
