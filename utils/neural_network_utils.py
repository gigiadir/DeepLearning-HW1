from activation import Activation
from layers.layer import Layer
from layers.res_net_layer import ResNetLayer
from layers.softmax_layer import SoftmaxLayer
from neural_network import NeuralNetwork
from tests.gradient_and_jacobian_test import gradient_test


def create_basic_neural_network_with_l_layers(l, input_dim, output_dim):
    layers = [Layer(input_dim = input_dim, output_dim = input_dim, activation=Activation.TANH) for _ in range(l-1)]
    loss_layer = SoftmaxLayer(input_dim=input_dim, output_dim = output_dim)

    nn_layers = layers + [loss_layer]
    nn = NeuralNetwork(nn_layers)

    return nn

def create_neural_network_with_l_resnet_layers(l, input_dim, output_dim):
    layers = [ResNetLayer(dim=input_dim, activation=Activation.TANH) for _ in range(l)]
    loss_layer = SoftmaxLayer(input_dim=input_dim, output_dim=output_dim)

    nn_layers = layers + [loss_layer]
    nn = NeuralNetwork(nn_layers)

    return nn

def create_neural_network_with_mixed_layers(n_resnet, n_layers, input_dim, output_dim):
    resnet_layers = [ResNetLayer(dim=input_dim, activation=Activation.TANH) for _ in range(n_resnet)]
    layers = [Layer(input_dim=input_dim, output_dim=input_dim, activation=Activation.TANH) for _ in range(n_layers)]
    loss_layer = SoftmaxLayer(input_dim=input_dim, output_dim=output_dim)

    nn_layers = resnet_layers + layers + [loss_layer]
    nn = NeuralNetwork(nn_layers)

    return nn

