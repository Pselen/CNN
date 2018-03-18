import numpy as np
from AbstractLayer import AbstractLayer

class ConvolutionLayer(AbstractLayer):
    def __init__(self, fshape, activation, filter_init, strides=1):
        self.fshape = fshape
        self.strides = strides
        self.filters = filter_init(self.fshape)
        self.activation = activation

    def forward(self, inputs):
        s = (inputs.shape[1] - self.fshape[0]) // self.strides + 1
        fmap = np.zeros((inputs.shape[0], s, s, self.fshape[-1]))
        for j in range(s):
            for i in range(s):
                fmap[:, j, i, :] = np.sum(inputs[:, j * self.strides:j * self.strides + self.fshape[0], i * self.strides:i * self.strides + self.fshape[1], :, np.newaxis] * self.filters, axis=(1, 2, 3))
        return self.activation.compute(fmap)

    def train_forward(self, inputs):
        s = (inputs.shape[1] - self.fshape[0]) // self.strides + 1
        fmap = np.zeros((inputs.shape[0], s, s, self.fshape[-1]))
        for j in range(s):
            for i in range(s):
                fmap[:, j, i, :] = np.sum(inputs[:, j * self.strides:j * self.strides + self.fshape[0], i * self.strides:i * self.strides + self.fshape[1], :, np.newaxis] * self.filters, axis=(1, 2, 3))
        return (fmap, self.activation.compute(fmap))

    def get_layer_error(self, z, backwarded_error):
        return backwarded_error * self.activation.derivative(z)

    def backward(self, layer_error):
        bfmap_shape = (layer_error.shape[1] - 1) * self.strides + self.fshape[0]
        backwarded_fmap = np.zeros((layer_error.shape[0], bfmap_shape, bfmap_shape, self.fshape[-2]))
        s = (backwarded_fmap.shape[1] - self.fshape[0]) // self.strides + 1
        for j in range(s):
            for i in range(s):
                backwarded_fmap[:, j * self.strides:j  * self.strides + self.fshape[0], i * self.strides:i * self.strides + self.fshape[1]] += np.sum(self.filters[np.newaxis, ...] * layer_error[:, j:j+1, i:i+1, np.newaxis, :], axis=4)
        return backwarded_fmap

    def get_gradient(self, inputs, layer_error):
        total_layer_error = np.sum(layer_error, axis=(0, 1, 2))
        filters_error = np.zeros(self.fshape)
        s = (inputs.shape[1] - self.fshape[0]) // self.strides + 1
        summed_inputs = np.sum(inputs, axis=0)
        for j in range(s):
            for i in range(s):
                filters_error += summed_inputs[j  * self.strides:j * self.strides + self.fshape[0], i * self.strides:i * self.strides + self.fshape[1], :, np.newaxis]
        return filters_error * total_layer_error

    def update(self, gradient):
        self.filters -= gradient
