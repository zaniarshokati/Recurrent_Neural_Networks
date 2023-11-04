import numpy as np
from Layers.Base import BaseLayer


class Pooling(BaseLayer):
    def __init__(self, stride_shape, pooling_shape, testing_phase=False):
        super().__init__(testing_phase)
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape
        self.str_h = stride_shape[0]
        self.str_w = stride_shape[1]
        self.p_h = pooling_shape[0]
        self.p_w = pooling_shape[1]

        self.input_tensor = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        batch, channel, height, width = input_tensor.shape
        output_sz0 = (height - self.p_h) // self.str_h + 1
        output_sz1 = (width - self.p_w) // self.str_w + 1
        output = np.zeros((batch, channel, output_sz0, output_sz1))
        for b in range(batch):
            for c in range(channel):
                for h in range(output_sz0):
                    for w in range(output_sz1):
                        output[b, c, h, w] = np.max(input_tensor[b, c, h*self.str_h:h*self.str_h+self.p_h,
                                                    w*self.str_w:w*self.str_w+self.p_w])
        return output

    def backward(self, error_tensor):
        batch, channel, height, width = self.input_tensor.shape
        output_sz0 = (height - self.p_h) // self.str_h + 1
        output_sz1 = (width - self.p_w) // self.str_w + 1
        output = np.zeros_like(self.input_tensor)
        for b in range(batch):
            for c in range(channel):
                for h in range(output_sz0):
                    for w in range(output_sz1):
                        sub_input_tensor = self.input_tensor[b, c,
                                     h * self.str_h:h * self.str_h + self.p_h,
                                     w * self.str_w:w * self.str_w + self.p_w]
                        mask = sub_input_tensor == np.max(sub_input_tensor)
                        usful_error = mask * error_tensor[b, c, h, w]
                        output[b, c, h * self.str_h:h * self.str_h + self.p_h, w * self.str_w:w * self.str_w + self.p_w] \
                            += usful_error #we put "+=" because selected windows have potentially overlap to deal with overlappin kernels
        return output