import numpy as np


class Optimizer:
    def __init__(self, learning_rate=0.01, regularizer=None):
        self.learning_rate = learning_rate
        self.regularizer = regularizer

    def add_regularizer(self, regularizer):
        self.regularizer = regularizer


class Sgd(Optimizer):
    def __init__(self, learning_rate=0.01, regularizer=None):
        super().__init__(learning_rate, regularizer)
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        new_weight = weight_tensor - self.learning_rate * gradient_tensor
        if self.regularizer is None:
            return new_weight
        else:
            shrinkage = self.regularizer.calculate_gradient(weight_tensor)
            return new_weight - self.learning_rate * shrinkage


class SgdWithMomentum(Optimizer):
    def __init__(self, learning_rate, momentum_rate, regularizer=None):
        super().__init__(learning_rate, regularizer)
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.v = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        self.v = (self.momentum_rate * self.v) - (self.learning_rate * gradient_tensor)
        new_weight = weight_tensor + self.v

        if self.regularizer is None:
            return new_weight
        else:
            shrinkage = self.regularizer.calculate_gradient(weight_tensor)
            return new_weight - self.learning_rate * shrinkage


class Adam(Optimizer):
    def __init__(self, learning_rate, mu, rho, regularizer=None):
        super().__init__(learning_rate, regularizer)
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.v = 0
        self.r = 0
        self.k_iterCounter = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        self.v = (self.mu * self.v) + (1 - self.mu) * gradient_tensor
        self.r = (self.rho * self.r) + (1 - self.rho) * np.square(gradient_tensor)
        self.k_iterCounter = self.k_iterCounter+1
        v_hat = self.v / (1 - self.mu ** self.k_iterCounter)
        r_hat = self.r / (1 - self.rho ** self.k_iterCounter)
        new_weight = weight_tensor - self.learning_rate * v_hat / (np.sqrt(r_hat) + np.finfo(float).eps)

        if self.regularizer is None:
            return new_weight
        else:
            shrinkage = self.regularizer.calculate_gradient(weight_tensor)
            return new_weight - self.learning_rate * shrinkage

