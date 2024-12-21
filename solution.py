import numpy as np
from matplotlib import pyplot as plt

from activation import Activation
from layers.layer import Layer
from layers.res_net_layer import ResNetLayer
from layers.softmax_layer import SoftmaxLayer
from neural_network import NeuralNetwork
from utils.data_utils import get_dataset, sample_minibatch
from gradient_descent import sgd, sgd_with_momentum
from tests.gradient_and_jacobian_test import gradient_test, jacobian_test, jacobian_transpose_test
from least_squares import generate_least_squares, least_squares_gradient, least_squares_loss, plot_line_fitting
from softmax_cross_entropy import softmax_cross_entropy_loss, softmax_cross_entropy_gradient_dw
from tests.sgd_test import compare_different_batch_sizes
from utils.figure_utils import plot_loss_over_iterations
from utils.neural_network_utils import create_basic_neural_network_with_l_layers


def section_1a():
    peaks_training_data, _ = get_dataset("Peaks", 0.25)
    X, C, n, l = peaks_training_data.X, peaks_training_data.C, peaks_training_data.n, peaks_training_data.l

    softmax_cross_entropy_func = lambda w : softmax_cross_entropy_loss(X, C, w)
    softmax_cross_entropy_gradient_func = lambda w : softmax_cross_entropy_gradient_dw(X, C, w)

    gradient_test(softmax_cross_entropy_func, softmax_cross_entropy_gradient_func, (n+1) * l, plot_title="Gradient Test Results - Cross Entropy Gradient")

def section_1b():
    m = 100
    A, b, x_points = generate_least_squares(m)
    x = np.random.rand(2, 1)
    _, sgd_loss_list, _, sgd_theta_list = sgd(A, b, x,
                                              grad_f=least_squares_gradient,
                                              loss_func=least_squares_loss,
                                              batch_size=10,
                                              max_iterations=300,
                                              tolerance=0.01)

    _, sgd_with_momentum_loss_list, _, sgd_with_momentum_theta_list = sgd_with_momentum(A, b, x,
                                                            grad_f=least_squares_gradient,
                                                            loss_func=least_squares_loss,
                                                            batch_size=10,
                                                            max_iterations=300,
                                                            tolerance=0.01)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plot_loss_over_iterations(sgd_loss_list, "SGD")
    plt.subplot(1, 2, 2)
    plot_loss_over_iterations(sgd_with_momentum_loss_list, "SGD with Momentum")
    plt.tight_layout()
    plt.show()

    plot_line_fitting(A, b, theta_list=sgd_theta_list[0:150], final_theta=sgd_theta_list[-1], x_points = x_points, method="SGD", interval = 50)
    plot_line_fitting(A, b, theta_list=sgd_with_momentum_theta_list[0:150], final_theta=sgd_with_momentum_theta_list[-1], x_points = x_points, method="SGD With Momentum", interval = 50)

def section_1c():
    # test_dataset_learning_rates_and_batch_sizes("SwissRoll")
    # test_dataset_learning_rates_and_batch_sizes("GMM")
    # test_dataset_learning_rates_and_batch_sizes("Peaks")
    compare_different_batch_sizes("GMM")

def section_2a():
    peaks_training_data, _ = get_dataset("Peaks")
    X, C, n, l = peaks_training_data.X_raw, peaks_training_data.C, peaks_training_data.n, peaks_training_data.l
    ms_size = 10
    X, C = sample_minibatch(X, C, ms_size, True)

    layer = Layer(
        input_dim=n,
        output_dim=l,
        activation=Activation.TANH
    )

    layer.forward(X=X, C=C)

    jacobian_test(lambda b: Layer(input_dim=n, output_dim=l, activation=Activation.TANH, b=b, w_vector=layer.w_vector).forward(X=X, C=C), layer.jac_db_mul_v, l)
    jacobian_test(lambda x: Layer(input_dim=n, output_dim=l, activation=Activation.TANH, b=layer.b, w_vector=layer.w_vector).forward(X=x.reshape(n, ms_size, order='F'), C=C), layer.jac_dx_mul_v, n*ms_size)
    jacobian_test(lambda w_vector: Layer(input_dim=n, output_dim=l, activation=Activation.TANH, b=layer.b, w_vector=w_vector).forward(X=X, C=C), layer.jac_dw_mul_v, l * n)


    X, C = sample_minibatch(X, C, 1, True)
    layer.forward(X=X, C=C)
    jacobian_transpose_test(layer.jac_db_mul_v, layer.jac_transpose_db_mul_v, l, l)
    jacobian_transpose_test(layer.jac_dx_mul_v, layer.jac_transpose_dx_mul_v, l, n)
    jacobian_transpose_test(layer.jac_dw_mul_v, layer.jac_transpose_dw_mul_v, l, n*l)

def section_2b():
    peaks_training_data, peaks_validation_data = get_dataset("Peaks")
    X_train, C_train, X_validation, C_validation = peaks_training_data.X_raw, peaks_training_data.C, peaks_validation_data.X_raw, peaks_validation_data.C
    X, C, n, l = peaks_training_data.X_raw, peaks_training_data.C, peaks_training_data.n, peaks_training_data.l
    ms_size = 10
    X, C = sample_minibatch(X, C, ms_size, True)

    resnet_layer = ResNetLayer(
        dim=n,
        activation=Activation.TANH
    )

    resnet_layer.forward(X=X, C=C)

    jacobian_test(lambda b: ResNetLayer(dim=n, activation=Activation.TANH,
                                        w1_vector = resnet_layer.w1_vector, w2_vector = resnet_layer.w2_vector,
                                        b=b).forward(X=X, C=C), resnet_layer.jac_db_mul_v, n)
    jacobian_test(lambda w1_vector: ResNetLayer(dim=n, activation=Activation.TANH,
                                        w1_vector=w1_vector, w2_vector=resnet_layer.w2_vector,
                                        b=resnet_layer.b).forward(X=X, C=C), resnet_layer.jac_dw1_mul_v, n * n)
    jacobian_test(lambda w2_vector: ResNetLayer(dim=n, activation=Activation.TANH,
                                                w1_vector=resnet_layer.w1_vector, w2_vector=w2_vector,
                                                b=resnet_layer.b).forward(X=X, C=C), resnet_layer.jac_dw2_mul_v, n * n)
    jacobian_test(lambda x_vector: resnet_layer.forward(X=x_vector.reshape(n, ms_size, order='F'), C=C), resnet_layer.jac_dx_mul_v,
                  n * ms_size)

    X, C = sample_minibatch(X, C, 1, True)
    resnet_layer.forward(X=X, C=C)
    jacobian_transpose_test(resnet_layer.jac_db_mul_v, resnet_layer.jac_transpose_db_mul_v, n, n)
    jacobian_transpose_test(resnet_layer.jac_dw1_mul_v, resnet_layer.jac_transpose_dw1_mul_v, n, n * n)
    jacobian_transpose_test(resnet_layer.jac_dw2_mul_v, resnet_layer.jac_transpose_dw2_mul_v, n, n * n)
    jacobian_transpose_test(resnet_layer.jac_dx_mul_v, resnet_layer.jac_transpose_dx_mul_v, n, n)

    nn = NeuralNetwork(
        layers=[
            ResNetLayer(dim=n,
                        activation=Activation.TANH),
            SoftmaxLayer(input_dim=n, output_dim=l)
        ]
    )
    nn.forward(X_train, C_train)
    nn.backprop()

    f = lambda weights_and_bias_vector: nn.set_weights_and_get_loss(weights_and_bias_vector, X=X_train, C=C_train)
    grad_f = lambda *args: nn.get_gradient_vector()

    x = nn.get_weights_and_biases_vector()

    gradient_test(f, grad_f, x.size, x)

def section_2c():
    L = 5
    peaks_training_data, peak_validation_data = get_dataset("Peaks")
    X_train, C_train, n, l = peaks_training_data.X_raw, peaks_training_data.C, peaks_training_data.n, peaks_training_data.l

    nn = create_basic_neural_network_with_l_layers(l=L, input_dim=n, output_dim=l)
    nn.forward(X_train, C_train)
    nn.backprop()

    f = lambda weights_and_bias_vector: nn.set_weights_and_get_loss(weights_and_bias_vector, X=X_train, C=C_train)
    grad_f = lambda *args: nn.get_gradient_vector()

    x = nn.get_weights_and_biases_vector()

    gradient_test(f, grad_f, x.size, x)

def section_2d():
    peaks_training_data, peak_validation_data = get_dataset("GMM")
    X_raw, C, n, l = peaks_training_data.X_raw, peaks_training_data.C, peaks_training_data.n, peaks_training_data.l
    X_validation, C_validation = peak_validation_data.X_raw, peak_validation_data.C

    nn = NeuralNetwork(
        layers = [
            Layer(
                input_dim = n,
                output_dim = 8,
                activation=Activation.TANH
            ),
            Layer(
                input_dim=8,
                output_dim=16,
                activation=Activation.TANH
            ),
            SoftmaxLayer(
                input_dim= 16,
                output_dim = l
            )
        ]
    )

    nn.train(X_train = X_raw,
             C_train = C,
             X_validation=X_validation,
             C_validation=C_validation,
             epochs=200,
             mb_size = 32,
             learning_rate = 0.05)

def section_1():
    section_1a()
    section_1b()
    section_1c()

def section_2():
    section_2a()
    section_2b()
    section_2c()
    section_2d()



if __name__ == '__main__':
    section_1c()