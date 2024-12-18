from typing import List

from layers.layer import Layer


class NeuralNetwork:
    def __init__(self, layers: List[Layer], X, C):
        self.layers = layers
        self.X = X
        self.C = C
        self.learning_rate = 0.01
        self.epochs = 1000

    def _get_minibatch(self):
        return self.X, self.C

    def train(self, epochs):
        epoch = 0
        while epoch < epochs:
            X, C = self._get_minibatch()
            self.forward(X, C)
            self.backprop()
            self.update_weights()

    def forward(self, X, C):
        next_X, next_C = X, C
        for layer in self.layers:
            next_X, next_C = layer.forward(next_X, next_C)

    def backprop(self):
        next_grad_x = None
        for layer in reversed(self.layers):
            next_grad_x = layer.backward(next_grad_x)

    def update_weights(self):
        for layer in self.layers:
            layer.update_parameters(self.learning_rate)
