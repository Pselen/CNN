import numpy as np
from collections import deque

class CNNetwork(object):
    def __init__(self, layers, learning_rate, loss_function):
        self.layers = layers
        self.loss_function = loss_function
        self.learning_rate = learning_rate

    def forward(self, inputs):
        activation = inputs
        for l in self.layers:
            activation = l.forward(activation)
        return activation

    def train_step(self, mini_batch):
        mini_batch_inputs, mini_batch_outputs = mini_batch
        scoring_values = deque([mini_batch_inputs])
        activation = mini_batch_inputs
        for l in self.layers:
            scoring_function, activation = l.train_forward(activation)
            scoring_values.appendleft(scoring_function)

        loss_error = self.loss_function.derivative((activation, mini_batch_outputs))
        lz = scoring_values.popleft()
        backwarded_error = loss_error
        gradients = deque()
        for l in reversed(self.layers):
            layer_error = l.get_layer_error(lz, backwarded_error) #local
            lz = scoring_values.popleft()
            gradients.appendleft(l.get_gradient(lz, layer_error))
            backwarded_error = l.backward(layer_error) # backwarded error

        # update step
        for l in self.layers:
            l.update(self.learning_rate * gradients.popleft())

        assert len(gradients) == 0
