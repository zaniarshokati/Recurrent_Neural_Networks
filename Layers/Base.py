# import copy
#
# class BaseLayer:
#     def __init__(self, testing_phase=False):
#         self.testing_phase = testing_phase
#         self._weights = 0
#         self._optimizer = None
#         self._bias_optimizer = None
#         self._gradient_weights = None
#         self._gradient_bias = None
#         self.input_tensor = None
#
#     @property
#     def optimizer(self):
#         return self._optimizer
#
#     @optimizer.setter
#     def optimizer(self, opt):
#         self._optimizer = opt
#         self._bias_optimizer = copy.copy(opt)
#
#     @property
#     def weights(self):
#         return self._weights
#
#     @weights.setter
#     def weights(self, w):
#         self._weights = w
#
#     @property
#     def gradient_weights(self):
#         return self._gradient_weights
#
#     @gradient_weights.setter
#     def gradient_weights(self, gw):
#         self._gradient_weights = gw
#
#     @property
#     def gradient_bias(self):
#         return self._gradient_bias
#
#     @gradient_bias.setter
#     def gradient_bias(self, gb):
#         self._gradient_bias = gb

import copy


class BaseLayer:
    def __init__(self, testing_phase=False):
        self.testing_phase = testing_phase
        self._weights = 0
        self._optimizer = None
        self._bias_optimizer = None
        self._gradient_weights = None
        self._gradient_bias = None
        self.input_tensor = None

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, opt):
        self._optimizer = opt
        self._bias_optimizer = copy.copy(opt)

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, w):
        self._weights = w

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, gw):
        self._gradient_weights = gw

    @property
    def gradient_bias(self):
        return self._gradient_bias

    @gradient_bias.setter
    def gradient_bias(self, gb):
        self._gradient_bias = gb
