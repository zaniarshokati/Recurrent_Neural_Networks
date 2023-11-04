import numpy as np
from Layers.Base import BaseLayer


class Constant(BaseLayer):
    def __init__(self, constant_value=.1):
        self.constant_value = constant_value

    def initialize(self, weights_shape, fan_in, fan_out):
        weights = np.ones(weights_shape) * self.constant_value
        return weights


class UniformRandom(BaseLayer):
    def __init__(self):
        pass
    def initialize(self, weights_shape, fan_in, fan_out):
        weights = np.random.uniform(0, 1, weights_shape)
        return weights


class Xavier(BaseLayer):
    def __init__(self):
        pass
    def initialize(self, weights_shape, fan_in, fan_out):
        sigma = np.sqrt(2 / (fan_out + fan_in))
        weights = np.random.normal(0, sigma, weights_shape)
        return weights


class He(BaseLayer):
    def __init__(self):
        pass
    def initialize(self, weights_shape, fan_in, fan_out):
        sigma = np.sqrt(2 / fan_in)
        weights = np.random.normal(loc=0, scale=sigma, size=weights_shape)
        return weights


