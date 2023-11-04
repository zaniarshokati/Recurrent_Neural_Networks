import numpy as np
from scipy.signal import correlate
from scipy.signal import convolve
import copy
from Layers.Base import BaseLayer


class Conv(BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels, testing_phase=False):
        super().__init__(testing_phase)
        #  stride Info
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels

        # convolution Info
        self.conv_c = self.convolution_shape[0]
        self.conv_m = self.convolution_shape[1]
        if np.size(self.convolution_shape) == 2:
            shape_weights = (num_kernels, self.conv_c, self.conv_m)
        else:
            self.conv_n = self.convolution_shape[2]
            shape_weights = (num_kernels, self.conv_c, self.conv_m, self.conv_n)

        # Padding Info
        self.s_y = self.stride_shape[0]
        pady1 = int(np.ceil((shape_weights[2] - 1) / 2))
        pady2 = int((shape_weights[2] - 1) / 2)
        if np.size(self.stride_shape) == 1:
            self.pad_shape = ((0, 0), (0, 0), (pady1, pady2))
        else:
            self.s_x = self.stride_shape[1]
            padx1 = int(np.ceil((shape_weights[3] - 1) / 2))  # shape_weights[3]//2
            padx2 = int((shape_weights[3] - 1) / 2)  # shape_weights[3] // 2
            self.pad_shape = ((0, 0), (0, 0), (pady1, pady2), (padx1, padx2))

        self.weights = np.random.uniform(0, 1, shape_weights)
        self.bias = np.random.uniform(0, 1, size=num_kernels)

        self.gradient_weights = None
        self.gradient_bias = None
        self.input_tensor = None
        self.optimizer = None
        self.bias_optimizer = None

        # gradient wrt weights property
        @property
        def gradient_weights(self):
            return self.gradient_weights

        @gradient_weights.setter
        def gradient_weights(self, gradient_weights):
            self.gradient_weights = gradient_weights

        # gradient wrt bias property
        @property
        def gradient_bias(self):
            return self.gradient_bias

        @gradient_bias.setter
        def gradient_bias(self, gradient_bias):
            self.gradient_bias = gradient_bias

        @property
        def optimizer(self):
            return self.optimizer

        @optimizer.setter
        def optimizer(self, optimizer):
            self.optimizer = optimizer
            self.bias_optimizer = copy.copy(optimizer)

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        if np.size(input_tensor.shape) == 4:
            b, c, y, x = input_tensor.shape
            output_shape = (b, self.num_kernels, 1 + (y - 1) // self.s_y, 1 + (x - 1) // self.s_x)
        else:
            b, c, y = input_tensor.shape
            output_shape = (b, self.num_kernels, 1 + (y - 1) // self.s_y)
        output = np.zeros(output_shape)
        input_tensor = np.pad(input_tensor, self.pad_shape, mode='constant')

        for batch_idx in range(b):
            for nk in range(self.num_kernels):
                corr_res = correlate(input_tensor[batch_idx], self.weights[nk], mode='valid') + self.bias[nk]
                if np.size(corr_res.shape) == 2:
                    res_ = corr_res[:, ::self.s_y]
                elif np.size(corr_res.shape) == 3:
                    res_ = corr_res[:, ::self.s_y, ::self.s_x]
                    # res_ = res_[0]
                output[batch_idx, nk, ...] = res_
        return output

    def backward(self, error_tensor):
        if np.size(error_tensor.shape) == 4:
            b, num_kernels, y_error, x_error = error_tensor.shape
            error_tensor_unsampled_shape = (b, num_kernels, self.input_tensor.shape[2], self.input_tensor.shape[3])
            error_tensor_unsampled = np.zeros(error_tensor_unsampled_shape)
            error_tensor_unsampled[:, :, ::self.s_y, ::self.s_x] = error_tensor
        else:
            b, num_kernels, y_error = error_tensor.shape
            error_tensor_unsampled_shape = (b, num_kernels, self.input_tensor.shape[2])
            error_tensor_unsampled = np.zeros(error_tensor_unsampled_shape)
            error_tensor_unsampled[:, :, ::self.s_y] = error_tensor

        error_tensor_padded = np.pad(error_tensor_unsampled, self.pad_shape, mode='constant')

        output = np.zeros_like(self.input_tensor)
        for channel in range(self.conv_c):
            error_kernel = self.weights[:, channel, ...]
            error_kernel = np.flip(error_kernel, axis=0)
            for batch_idx in range(b):
                conv_res = convolve(error_tensor_padded[batch_idx, ...], error_kernel, mode='valid')

                if np.size(conv_res.shape) == 2:
                    output[batch_idx, channel, :] = conv_res
                elif np.size(conv_res.shape) == 3:
                    output[batch_idx, channel, :, :] = conv_res

        input_padded = np.pad(self.input_tensor, self.pad_shape, mode='constant')
        self.gradient_bias = np.zeros_like(self.bias)
        self.gradient_weights = np.zeros_like(self.weights)

        if np.size(error_tensor.shape) == 4:
            self.gradient_bias = np.sum(error_tensor, axis=(0, 2, 3))#skip kernels since every kernel has its own bias
        else:
            self.gradient_bias = np.sum(error_tensor, axis=(0, 2))#skip kernels

        for batch_idx in range(b):
            for nk in range(num_kernels):
                temp_1 = input_padded[batch_idx, ...]
                temp_2 = error_tensor_unsampled[np.newaxis, batch_idx, nk, ...]
                res_grad = correlate(temp_1, temp_2, mode='valid')
                self.gradient_weights[nk, ...] += res_grad

        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)
        if self.bias_optimizer is not None:
            self.bias = self.bias_optimizer.calculate_update(self.bias, self.gradient_bias)

        return output

    def initialize(self, weight_initializer, bias_initializer):
        if np.size(self.convolution_shape) == 2:
            fan_in = self.conv_c * self.conv_m
            fan_out = self.num_kernels * self.conv_c * self.conv_m
        else:
            fan_in = self.conv_c * self.conv_m * self.conv_n
            fan_out = self.num_kernels * self.conv_m * self.conv_n

        self.weights = weight_initializer.initialize(self.weights.shape, fan_in, fan_out)
        self.bias = bias_initializer.initialize(self.bias.shape, fan_in, fan_out)