from typing import List, Union
import numpy as np

from layers.layer import Layer
from layers.res_net_layer import ResNetLayer
from layers.softmax_layer import SoftmaxLayer


class NeuralNetwork:
    def __init__(self, layers: List[Union[Layer, ResNetLayer, SoftmaxLayer]]):
        self.layers = layers
        self.layers_weights_and_biases_vectors = []


    def train(self, X_train, C_train, X_validation, C_validation, epochs, mb_size, learning_rate):
        epoch = 0
        num_samples = X_train.shape[1]
        num_minibatches = int(num_samples / mb_size)
        train_accuracy, validation_accuracy, loss_list = [], [], []
        while epoch < epochs:
            shuffled_indices = np.random.permutation(num_samples)
            loss = []
            lr = learning_rate * (320/(320 + epoch))
            for i in range(0, num_samples, mb_size):
                indices = shuffled_indices[i:min(i + mb_size, num_samples)]
                X_mb, C_mb = X_train[:,indices], C_train[:,indices]
                loss += self.forward(X=X_mb, C=C_mb)
                self.backprop()
                self.update_weights(lr)

            train_acc = self.test(X_train, C_train)
            train_accuracy.append(train_acc)
            validation_acc = self.test(X_validation, C_validation)
            validation_accuracy.append(validation_acc)
            loss_list.append(loss / num_minibatches)

            print(f"epoch {epoch} finished. accuracy on validation set: {validation_acc}. accuracy on training set: {train_acc}")
            epoch += 1

        return train_accuracy, validation_accuracy, loss_list

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

    def get_gradient_vector(self):
        gradients = []
        for layer in self.layers:
            gradients.append(layer.get_gradient_vector())

        gradients_vector = np.concatenate(gradients, axis=0)
        return gradients_vector

    def get_weights_and_biases_vector(self):
        weights_and_biases_vector = []
        for layer in self.layers:
            weights_and_biases_vector.append(layer.get_weights_and_bias_vector())

        self.layers_weights_and_biases_vectors = weights_and_biases_vector

        weights_and_biases_vector = np.concatenate(weights_and_biases_vector, axis=0)

        return weights_and_biases_vector

    def set_weights_and_biases_vector(self, vector):
        index = 0

        for layer in self.layers:
            layer_vector_size = layer.get_weights_and_bias_vector().size
            layer_vector = vector[index:index + layer_vector_size]
            layer.set_weights_and_bias_from_vector(layer_vector)
            index += layer_vector_size

    def set_weights_and_get_loss(self, weights_and_biases_vector, X, C):
        self.set_weights_and_biases_vector(weights_and_biases_vector)
        loss = self.forward(X=X, C=C)

        return loss
