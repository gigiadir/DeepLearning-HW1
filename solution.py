import numpy as np

from activation import Activation
from layers.layer import Layer
from layers.res_net_layer import ResNetLayer
from layers.softmax_layer import SoftmaxLayer
from neural_network import NeuralNetwork
from utils.data_utils import get_dataset, sample_minibatch
from gradient_descent import sgd
from tests.gradient_and_jacobian_test import gradient_test, jacobian_test, jacobian_transpose_test
from least_squares import generate_least_squares, least_squares_gradient, plot_gradient_descent_least_squares_result, \
    least_squares_loss
from softmax_cross_entropy import softmax_cross_entropy_loss, softmax_cross_entropy_gradient_dw
from tests.sgd_test import test_dataset_learning_rates_and_batch_sizes


def section_1a():
    peaks_training_data, _ = get_dataset("Peaks")
    X, C, n, l = peaks_training_data.X, peaks_training_data.C, peaks_training_data.n, peaks_training_data.l

    softmax_cross_entropy_func = lambda w : softmax_cross_entropy_loss(X, C, w)
    softmax_cross_entropy_gradient_func = lambda w : softmax_cross_entropy_gradient_dw(X, C, w)

    gradient_test(softmax_cross_entropy_func, softmax_cross_entropy_gradient_func, (n+1) * l)

def section_1b():
    m = 20
    A, b, x_points = generate_least_squares(m)
    x = np.random.rand(2, 1)
    min_point, loss_list, _, theta_list = sgd(A, b, x,
                                              grad_f=least_squares_gradient,
                                              loss_func=least_squares_loss,
                                              accuracy_func=lambda *args: args,
                                              batch_size=5,
                                              max_epochs=400,
                                              tolerance=-0.01)
    for i in range(0, len(theta_list), 30):
        plot_gradient_descent_least_squares_result(x_points, b, theta_list[i])


def section_1c():
    test_dataset_learning_rates_and_batch_sizes("Peaks")
    test_dataset_learning_rates_and_batch_sizes("GMM")
    test_dataset_learning_rates_and_batch_sizes("SwissRoll")

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
    peaks_training_data, _ = get_dataset("Peaks")
    X, C, n, l = peaks_training_data.X_raw, peaks_training_data.C, peaks_training_data.n, peaks_training_data.l
    ms_size = 10
    X, C = sample_minibatch(X, C, ms_size, True)

    layer = ResNetLayer(
        dim=n,
        activation=Activation.TANH
    )

    layer.forward(X=X, C=C)
    jacobian_test(lambda b: layer.forward(X=X, b=b)[0], layer.jac_db_mul_v, l)
    jacobian_test(lambda x: layer.forward(X=x.reshape(n, ms_size, order='F'))[0], layer.jac_dx_mul_v, n * ms_size)
    jacobian_test(lambda w_vector: layer.forward(X=X, W=w_vector.reshape(l, n, order='F'))[0], layer.jac_dw_mul_v,
                  l * n)

    X, C = sample_minibatch(X, C, 1, True)
    layer.forward(X=X, C=C)
    jacobian_transpose_test(layer.jac_db_mul_v, layer.jac_transpose_db_mul_v, n, n)
    jacobian_transpose_test(layer.jac_dx_mul_v, layer.jac_transpose_dx_mul_v, n, n)
    jacobian_transpose_test(layer.jac_dw_mul_v, layer.jac_transpose_dw_mul_v, n, n * n)

def section_2c():
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
                output_dim=32,
                activation=Activation.TANH
            ),
            Layer(
                input_dim=32,
                output_dim=64,
                activation=Activation.TANH
            ),
            Layer(
                input_dim=64,
                output_dim=128,
                activation=Activation.TANH
            ),
            Layer(
                input_dim=128,
                output_dim=64,
                activation=Activation.TANH
            ),
            Layer(
                input_dim=64,
                output_dim=32,
                activation=Activation.TANH
            ),
            Layer(
                input_dim=32,
                output_dim=16,
                activation=Activation.TANH
            ),
            SoftmaxLayer(
                input_dim= 16,
                output_dim = l
            )
        ],
        X = X_raw,
        C = C,
        X_validation=X_validation,
        C_validation=C_validation
    )

    nn.train(epochs=200, mb_size = 32, learning_rate = 0.1)

def section_1():
    section_1a()
    section_1b()
    section_1c()

def section_2():
    #section_2a()
    section_2c()



if __name__ == '__main__':
    #section_1()
    section_2()