import numpy as np
from . import TanH, Sigmoid, FullyConnected
from Layers.Base import BaseLayer


class RNN(BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden_state = None
        self.hidden_state_0 = np.zeros(hidden_size)
        self._memorize = False

        self.FCL_h = FullyConnected.FullyConnected(input_size + hidden_size, hidden_size)
        self.FCL_y = FullyConnected.FullyConnected(hidden_size, output_size)
        self.TanH = TanH.TanH()
        self.Sigmoid = Sigmoid.Sigmoid()

        self.batch_size = None
        self.hidden_in = None

    @property
    def memorize(self):
        return self._memorize

    @memorize.setter
    def memorize(self, m_flag):
        self._memorize = m_flag

    @property
    def weights(self):
        return self.FCL_h.weights

    @weights.setter
    def weights(self, w):
        self.FCL_h.weights = w

    def initialize(self, weights_initializer, bias_initializer):
        self.FCL_h.initialize(weights_initializer, bias_initializer)
        self.FCL_y.initialize(weights_initializer, bias_initializer)

    def forward(self, input_tensor):
        self.batch_size, input_size = input_tensor.shape
        self.hidden_state = np.zeros((self.batch_size + 1, self.hidden_size))
        if self.memorize:
            self.hidden_state[0, :] = self.hidden_state_0

        for t in range(self.batch_size):
            h_tm1 = self.hidden_state[t, :]#this is h_{t-1} shown in the slides
            X_t = input_tensor[t, :]
            x_tilda = np.concatenate((X_t, h_tm1))
            self.hidden_state[t + 1, :] = \
                self.FCL_h.forward(x_tilda.reshape(1, self.hidden_size + self.input_size))
            self.hidden_state[t + 1, :] = \
                self.TanH.forward(self.hidden_state[t + 1, :])

        self.hidden_state_0 = self.hidden_state[-1, :]#save the last for future
        self.hidden_in = np.concatenate((np.ones((self.batch_size, 1)),
                                         input_tensor, self.hidden_state[:-1, :]), axis=1)

        y = self.FCL_y.forward(self.hidden_state[1:, :])#there is no y_{t-1} from the start
        y = self.Sigmoid.forward(y)

        return y

    def backward(self, error_tensor):
        error_below_sigmoid = self.Sigmoid.backward(error_tensor)
        err_y = self.FCL_y.backward(error_below_sigmoid)
        gradient_hidden = np.zeros_like(self.hidden_state)#left wing in the intersection
        #Lets do the last cell
        gradient_hidden[-1, :] = err_y[-1, -self.hidden_size:]#last time point does not have err_h -> no right wing
        gradient_input = np.zeros((self.batch_size, self.input_size))
        self.gradient_weights = np.zeros_like(self.weights)
        shape_input_h = np.shape(self.hidden_in)
        for t in reversed(range(1, self.batch_size)):#We should deal only with the cells in between
            self.TanH.activations = self.hidden_state[t + 1, :]
            err_h_tanh = self.TanH.backward(gradient_hidden[t + 1, :])
            self.FCL_h.input_tensor = np.reshape(self.hidden_in[t, :], (1, shape_input_h[1]))
            err_h = self.FCL_h.backward(np.reshape(err_h_tanh, (1, len(err_h_tanh))))
            gradient_hidden[t, :] = err_y[t - 1, :] + err_h[:, -self.hidden_size:]#Sum both the the upper & right wings
            gradient_input[t, :] = err_h[:, :self.input_size]
            self.gradient_weights += self.FCL_h.gradient_weights
        #Only the first cell is remaining
        self.TanH.activations = self.hidden_state[1, :]
        err_h_tanh = self.TanH.backward(gradient_hidden[1, :])
        self.FCL_h.input_tensor = np.reshape(self.hidden_in[0, :], (1, shape_input_h[1]))
        err_h = self.FCL_h.backward(np.reshape(err_h_tanh, (1, len(err_h_tanh))))
        gradient_hidden[0, :] = err_h[:, -self.hidden_size:]#First time point does not have err_y -> no upper wing
        gradient_input[0, :] = err_h[:, :self.input_size]
        self.gradient_weights += self.FCL_h.gradient_weights

        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)
        return gradient_input

