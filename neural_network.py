from typing import List

import numpy as np

from layers.layer import Layer
from utils.data_utils import sample_minibatch


class NeuralNetwork:
    def __init__(self, layers: List[Layer], X, C, X_validation, C_validation):
        self.layers = layers
        self.X = X
        self.C = C
        self.X_validation = X_validation
        self.C_validation = C_validation

        self.weights = []


    def train(self, epochs, mb_size, learning_rate):
        epoch = 0
        num_samples = self.X.shape[1]

        while epoch < epochs:
            shuffled_indices = np.random.permutation(num_samples)
            loss = 0
            lr = learning_rate * (32/(32 + epoch))
            for i in range(0, num_samples, mb_size):
                indices = shuffled_indices[i:i + mb_size]
                X_mb, C_mb = self.X[:,indices], self.C[:,indices]
                loss += self.forward(X=X_mb, C=C_mb)
                self.backprop()
                self.update_weights(lr)

            train_accuracy = self.test(self.X, self.C)
            validation_accuracy = self.test(self.X_validation, self.C_validation)

            print(f"epoch {epoch} finished. accuracy on validation set: {validation_accuracy}. accuracy on training set: {train_accuracy}")
            epoch += 1

    def test(self, X_test, C_test):
        self.forward(X=X_test, C=C_test)
        classifications = np.argmax(self.softmax_probabilities, axis=1)
        true_labels = np.argmax(C_test, axis=0)
        correct = np.sum(classifications == true_labels)

        accuracy_percentage = correct / C_test.shape[1]
        return accuracy_percentage

    def forward(self, X, C):
        next_X = X
        for layer in self.layers:
            next_X = layer.forward(X=next_X, C=C)

        softmax_probabilities, loss = next_X
        self.softmax_probabilities = softmax_probabilities

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