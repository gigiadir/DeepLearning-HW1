from activation import Activation
from layers.layer import Layer
from layers.res_net_layer import ResNetLayer
from layers.softmax_layer import SoftmaxLayer
from neural_network import NeuralNetwork


def create_basic_neural_network_with_l_layers(l, input_dim, output_dim, n_neurons_each_layer = None, activation=None):
    activation = activation if activation is not None else Activation.TANH
    n_neurons_each_layer = n_neurons_each_layer if n_neurons_each_layer is not None else input_dim
    input_layer = Layer(input_dim = input_dim, output_dim = n_neurons_each_layer, activation=activation)
    hidden_layers = [Layer(input_dim = n_neurons_each_layer, output_dim = n_neurons_each_layer, activation=activation) for _ in range(l-2)]
    loss_layer = SoftmaxLayer(input_dim=n_neurons_each_layer, output_dim = output_dim)

    nn_layers = [input_layer] + hidden_layers + [loss_layer]
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

