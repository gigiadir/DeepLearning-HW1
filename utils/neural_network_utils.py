from activation import Activation
from layers.layer import Layer
from layers.softmax_layer import SoftmaxLayer
from neural_network import NeuralNetwork


def create_basic_neural_network_with_l_layers(l, input_dim, output_dim):
    layers = [Layer(input_dim = input_dim, output_dim = input_dim, activation=Activation.TANH) for _ in range(l-1)]
    loss_layer = SoftmaxLayer(input_dim=input_dim, output_dim = output_dim)

    nn_layers = layers + [loss_layer]
    nn = NeuralNetwork(nn_layers)

    return nn