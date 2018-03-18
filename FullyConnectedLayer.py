from AbstractLayer import AbstractLayer

class FullyConnectedLayer(AbstractLayer):
    def __init__(self, wshape, activation, weight_init):
        self.wshape = wshape
        self.weights = weight_init(self.wshape)
        self.activation = activation

    def forward(self, inputs):
        return self.activation.compute(inputs.dot(self.weights))

    def train_forward(self, inputs):
        scoring_values = inputs.dot(self.weights)
        return (scoring_values, self.activation.compute(scoring_values))

    def get_layer_error(self, scoring_values, backwarded_error):
        return backwarded_error * self.activation.derivative(scoring_values)

    def backward(self, layer_error):
        return layer_error.dot(self.weights.T)

    def get_gradient(self, inputs, layer_error):
        return inputs.T.dot(layer_error)

    def update(self, gradient):
        self.weights -= gradient
