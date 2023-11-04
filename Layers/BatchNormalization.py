from .Base import BaseLayer
from .Helpers import compute_bn_gradients
import numpy as np


class BatchNormalization(BaseLayer):
    def __init__(self, channels, optimizer=None, testing_phase=False):
        super().__init__(testing_phase)
        self.channels = channels
        self.weights = None  # gamma
        self.bias = None  # beta

        self.mean_tilda = 0
        self.var_tilda = 0

        self.mean_batch = None
        self.var_batch = None

        self.moving_average_alpha = 0.8
        self.optimizer = optimizer
        self.initialize()

        self.first_batch_flag = True
        self.input_tensor = None
        self.input_tilda = None
        self.shape_input_tensor = None

    def initialize(self, weights_initializer=None, bias_initializer=None):
        self.weights = np.ones(self.channels)
        self.bias = np.zeros(self.channels)

    def forward(self, input_tensor):
        self.shape_input_tensor = np.shape(input_tensor)
        if len(self.shape_input_tensor) > 2:  # reformat input_tensor to go from conv layer to fully i.e. 4D to 2D
            input_tensor = self.reformat(input_tensor)

        self.input_tensor = input_tensor

        if self.testing_phase:
            mean_batch = self.mean_tilda
            var_batch = self.var_tilda
        else:
            mean_batch = np.mean(input_tensor, axis=0)
            var_batch = np.var(input_tensor, axis=0)
            self.mean_tilda = self.moving_average_alpha * self.mean_tilda + (1 - self.moving_average_alpha) * mean_batch
            self.var_tilda = self.moving_average_alpha * self.var_tilda + (1 - self.moving_average_alpha) * var_batch
            self.mean_batch = mean_batch
            self.var_batch = var_batch

        input_tilda = (input_tensor - mean_batch) / np.sqrt(var_batch + np.finfo(float).eps)
        y_hat = self.weights * input_tilda + self.bias
        self.input_tilda = input_tilda

        if self.first_batch_flag:
            self.mean_tilda = mean_batch
            self.var_tilda = var_batch
            self.first_batch_flag = False

        if len(self.shape_input_tensor) > 2:
            y_hat = self.reformat(y_hat)

        return y_hat

    def backward(self, error_tensor):
        if len(self.shape_input_tensor) > 2:
            error_tensor = self.reformat(error_tensor)

        out = compute_bn_gradients(error_tensor, self.input_tensor, self.weights, self.mean_batch, self.var_batch)

        if len(self.shape_input_tensor) > 2:
            out = self.reformat(out)

        self._gradient_weights = np.sum(error_tensor * self.input_tilda, axis=0)
        self._gradient_bias = np.sum(error_tensor, axis=0)
        if self._optimizer is not None:
            self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)
            self.bias = self._bias_optimizer.calculate_update(self.bias, self._gradient_bias)
        return out

    def reformat(self, tensor):
        shape = np.shape(tensor)
        if len(shape) > 2:
            # Its time to convert conv layer into fully connected ones
            out = tensor.reshape(shape[0], shape[1], shape[2] * shape[3])
            out = np.transpose(out, axes=(0, 2, 1))
            out = out.reshape(out.shape[0] * out.shape[1], out.shape[2])
        else:
            # the shape is 2D now, but before it was 4D, lets go back to the original 4D format
            b, c, m, n = self.shape_input_tensor
            out = tensor.reshape(b, shape[0] // b, shape[1])
            out = np.transpose(out, axes=(0, 2, 1))
            out = out.reshape(b, c, m, n)
        return out