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


    def train(self, epochs, mb_size, learning_rate):
        epoch = 0
        num_samples = self.X.shape[1]
        num_minibatches = int(num_samples / mb_size)

        while epoch < epochs:
            shuffled_indices = np.random.permutation(num_samples)
            loss = 0
            for i in range(0, num_samples, mb_size):
                indices = shuffled_indices[i:i + mb_size]
                X_mb, C_mb = self.X[:,indices], self.C[:,indices]
                loss += self.forward(X=X_mb, C=C_mb)
                self.backprop()
                self.update_weights(learning_rate)

            print(f"epoch {epoch} finished. current loss- {loss/num_minibatches}")
            epoch += 1

    def test(self, X_test, C_test):
        self.forward(X=X_test, C=self.C)


    def forward(self, X, C):
        next_X = X
        for layer in self.layers:
            next_X = layer.forward(X=next_X, C=C)

        #The last layer forward returns the loss
        loss = next_X

        return loss
    def get_current_loss(self):
        loss_layer = self.layers[-1]
        loss = loss_layer.get_loss()

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