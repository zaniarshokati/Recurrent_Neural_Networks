import numpy as np


class CrossEntropyLoss:
    def __init__(self):
        self.y_hat = None

    def forward(self, input_tensor, label_tensor):
        self.y_hat = input_tensor
        mask = label_tensor == 1
        loss = np.sum(-1 * np.log(self.y_hat + np.finfo(float).eps) * mask)
        return loss

    def backward(self, label_tensor):
        error_tensor = np.divide(np.multiply(-1.0, label_tensor), self.y_hat)
        return error_tensor
