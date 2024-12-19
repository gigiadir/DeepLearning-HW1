from typing import List

import numpy as np

from layers.layer import Layer
from utils.data_utils import sample_minibatch


class NeuralNetwork:
    def __init__(self, layers: List[Layer], X, C):
        self.layers = layers
        self.X = X
        self.C = C
        self.weights = []


    def _get_minibatch(self, mb_size):
        X_mb, C_mb = sample_minibatch(self.X, self.C, mb_size, is_samples_in_columns=True)

        return X_mb, C_mb

    def train(self, epochs, mb_size, learning_rate):
        epoch = 0
        loss_list = []
        while epoch < epochs:
            X, C = self._get_minibatch(mb_size)
            loss = self.forward(X=X, C=C)
            loss_list.append(loss)
            if epoch % 20 == 0:
                print(f"epoch {epoch} - loss: {loss}")
            self.backprop()
            self.update_weights(learning_rate)
            epoch += 1

    def forward(self, X, C):
        next_X = X
        for layer in self.layers:
            next_X = layer.forward(X=next_X, C=C)

        #The lasy layer forward returns the loss
        loss = next_X

        return loss

    def backprop(self):
        next_grad_x = None
        for layer in reversed(self.layers):
            next_grad_x = layer.backprop(next_grad_x)

    def update_weights(self, learning_rate):
        for layer in self.layers:
            layer.update_weights(learning_rate)


    def get_weights(self):
        weights = []
        for layer in self.layers:
            weights.append(layer.get_weights())

        return weights