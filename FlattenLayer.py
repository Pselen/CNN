import numpy as np
from AbstractLayer import AbstractLayer

class FlattenLayer(AbstractLayer):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, inputs):
        return np.reshape(inputs, (inputs.shape[0], -1))

    def train_forward(self, inputs):
        scoring_values = np.reshape(inputs, (inputs.shape[0], -1))
        return (scoring_values, scoring_values)

    def get_layer_error(self, z, next_layer_error):
        return next_layer_error

    def backward(self, layer_error):
        return np.reshape(layer_error, (layer_error.shape[0],) + self.shape)

    def get_gradient(self, inputs, layer_error):
        return 0.

    def update(self, grad):
        pass
