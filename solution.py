import numpy as np
from matplotlib import pyplot as plt

from activation import Activation
from layers.layer import Layer
from layers.res_net_layer import ResNetLayer
from layers.softmax_layer import SoftmaxLayer
from neural_network import NeuralNetwork
from tests.neural_network_test import test_different_network_depths, test_different_learning_rates, \
    test_different_batch_sizes, test_network_for_swissroll
from utils.data_utils import get_dataset, sample_minibatch
from gradient_descent import sgd, sgd_with_momentum
from tests.gradient_and_jacobian_test import gradient_test, jacobian_test, jacobian_transpose_test, gradient_test_for_nn
from least_squares import generate_least_squares, least_squares_gradient, least_squares_loss, plot_line_fitting
from softmax_cross_entropy import softmax_cross_entropy_loss, softmax_cross_entropy_gradient_dw, \
    softmax_cross_entropy_accuracy
from tests.sgd_test import compare_different_batch_sizes, compare_different_learning_rates
from utils.figure_utils import plot_loss_over_iterations, plot_train_vs_validation_results
from utils.neural_network_utils import create_basic_neural_network_with_l_layers, \
    create_neural_network_with_l_resnet_layers, create_neural_network_with_mixed_layers
from utils.vector_utils import flatten_weights_matrix_to_vector


def section_1a():
    peaks_training_data, _ = get_dataset("Peaks", 0.25)
    X, C, n, l = peaks_training_data.X, peaks_training_data.C, peaks_training_data.n, peaks_training_data.l

    softmax_cross_entropy_func = lambda w : softmax_cross_entropy_loss(X, C, w)
    softmax_cross_entropy_gradient_func = lambda w : softmax_cross_entropy_gradient_dw(X, C, w)

    gradient_test(softmax_cross_entropy_func, softmax_cross_entropy_gradient_func, (n+1) * l,
                  plot_title="Gradient Test Results - Cross Entropy Gradient",
                  filename = "output/Section 1a/Gradient Test Results - Cross Entropy Gradient.png")

def section_1b():
    m = 200
    A, b, x_points = generate_least_squares(m)
    x = np.random.rand(2, 1)
    batch_size = 16
    max_iterations = 200

    _, sgd_loss_list, _, sgd_theta_list = sgd(A, b, x,
                                              grad_f=least_squares_gradient,
                                              loss_func=least_squares_loss,
                                              batch_size=batch_size,
                                              max_iterations=max_iterations,
                                              tolerance=0.01)

    _, sgd_with_momentum_loss_list, _, sgd_with_momentum_theta_list = sgd_with_momentum(A, b, x,
                                                                                        grad_f=least_squares_gradient,
                                                                                        loss_func=least_squares_loss,
                                                                                        batch_size=batch_size,
                                                                                        max_epochs=max_iterations,
                                                                                        tolerance=0.01)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plot_loss_over_iterations(sgd_loss_list, "SGD")
    plt.subplot(1, 2, 2)
    plot_loss_over_iterations(sgd_with_momentum_loss_list, "SGD with Momentum")
    plt.tight_layout()
    plt.show()
    plt.savefig(f"output/Section 1b/Least Squares Problem SGD vs SGD Momentum.png", dpi=300, bbox_inches='tight')

    interval = 50
    plot_line_fitting(A, b, theta_list=sgd_theta_list[0:150], final_theta=sgd_theta_list[-1], x_points = x_points, method="SGD", interval = interval)
    plot_line_fitting(A, b, theta_list=sgd_with_momentum_theta_list[0:150], final_theta=sgd_with_momentum_theta_list[-1], x_points = x_points, method="SGD With Momentum", interval = interval)


def section_1c():
    for dataset in ["SwissRoll"]:
    #     compare_different_batch_sizes(dataset)
    #     compare_different_learning_rates(dataset)
        dataset_training_data, dataset_validation_data = get_dataset(dataset, 1)
        X_train, C_train, W = dataset_training_data.X, dataset_training_data.C, dataset_training_data.W
        X_validation, C_validation = dataset_validation_data.X, dataset_validation_data.C
        initial_weights_vector = flatten_weights_matrix_to_vector(W)

        theta, loss_list, train_accuracy_list, theta_list = sgd_with_momentum(X_train, C_train, initial_weights_vector, grad_f=softmax_cross_entropy_gradient_dw,
                                    loss_func=softmax_cross_entropy_loss, accuracy_func=softmax_cross_entropy_accuracy,
                                    batch_size = 64, learning_rate=0.001, max_epochs=300)

        validation_accuracy_list = [softmax_cross_entropy_accuracy(X_validation, C_validation, theta) for theta in theta_list]
        plot_train_vs_validation_results(values_train=train_accuracy_list, values_validation=validation_accuracy_list, x_label="Epoch",
                                         title=f"{dataset} - Accuracy By Epoch")
        plot_train_vs_validation_results(loss_list, None, x_label="Epoch", y_label="Loss", title=f"{dataset} - Loss Function")





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

    jacobian_test(lambda b: Layer(input_dim=n, output_dim=l, activation=Activation.TANH, b=b, w_vector=layer.w_vector).forward(X=X, C=C), layer.jac_db_mul_v, l,
                  title=r'Jacobian Test Results - $\frac{\partial \mathbf{y}}{\partial \mathbf{b}}$')
    jacobian_test(lambda x: Layer(input_dim=n, output_dim=l, activation=Activation.TANH, b=layer.b, w_vector=layer.w_vector).forward(X=x.reshape(n, ms_size, order='F'), C=C), layer.jac_dx_mul_v, n*ms_size,
                  title=r'Jacobian Test Results - $\frac{\partial \mathbf{y}}{\partial \mathbf{x}}$')
    jacobian_test(lambda w_vector: Layer(input_dim=n, output_dim=l, activation=Activation.TANH, b=layer.b, w_vector=w_vector).forward(X=X, C=C), layer.jac_dw_mul_v, l * n,
                  title=r'Jacobian Test Results - $\frac{\partial \mathbf{y}}{\partial \mathbf{W}}$')


    X, C = sample_minibatch(X, C, 1, True)
    layer.forward(X=X, C=C)
    jacobian_transpose_test(layer.jac_db_mul_v, layer.jac_transpose_db_mul_v, l, l,
                            title=r'Jacobian Transpose Test Results - $\frac{\partial \mathbf{y^{T}}}{\partial \mathbf{b}}$')
    jacobian_transpose_test(layer.jac_dx_mul_v, layer.jac_transpose_dx_mul_v, l, n,
                            title=r'Jacobian Transpose Test Results - $\frac{\partial \mathbf{y^{T}}}{\partial \mathbf{x}}$')
    jacobian_transpose_test(layer.jac_dw_mul_v, layer.jac_transpose_dw_mul_v, l, n*l,
                            title=r'Jacobian Transpose Test Results - $\frac{\partial \mathbf{y^{T}}}{\partial \mathbf{W}}$')

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
                                        b=b).forward(X=X, C=C), resnet_layer.jac_db_mul_v, n,
                  title=r'Jacobian Test Results - $\frac{\partial \mathbf{y}}{\partial \mathbf{b}}$')
    jacobian_test(lambda w1_vector: ResNetLayer(dim=n, activation=Activation.TANH,
                                        w1_vector=w1_vector, w2_vector=resnet_layer.w2_vector,
                                        b=resnet_layer.b).forward(X=X, C=C), resnet_layer.jac_dw1_mul_v, n * n,
                  title=r'Jacobian Test Results - $\frac{\partial \mathbf{y}}{\partial \mathbf{W_1}}$')
    jacobian_test(lambda w2_vector: ResNetLayer(dim=n, activation=Activation.TANH,
                                                w1_vector=resnet_layer.w1_vector, w2_vector=w2_vector,
                                                b=resnet_layer.b).forward(X=X, C=C), resnet_layer.jac_dw2_mul_v, n * n,
                  title= r'Jacobian Test Results - $\frac{\partial \mathbf{y}}{\partial \mathbf{W_2}}$')
    jacobian_test(lambda x_vector: resnet_layer.forward(X=x_vector.reshape(n, ms_size, order='F'), C=C), resnet_layer.jac_dx_mul_v,
                  n * ms_size, title=r'Jacobian Test Results - $\frac{\partial \mathbf{y}}{\partial \mathbf{x}}$')

    X, C = sample_minibatch(X, C, 1, True)
    resnet_layer.forward(X=X, C=C)
    jacobian_transpose_test(resnet_layer.jac_db_mul_v, resnet_layer.jac_transpose_db_mul_v, n, n,
                            title=r'Jacobian Transpose Test Results - $\frac{\partial \mathbf{y^{T}}}{\partial \mathbf{b}}$')
    jacobian_transpose_test(resnet_layer.jac_dw1_mul_v, resnet_layer.jac_transpose_dw1_mul_v, n, n * n,
                            title=r'Jacobian Transpose Test Results - $\frac{\partial \mathbf{y^{T}}}{\partial \mathbf{W_1}}$')
    jacobian_transpose_test(resnet_layer.jac_dw2_mul_v, resnet_layer.jac_transpose_dw2_mul_v, n, n * n,
                            title=r'Jacobian Transpose Test Results - $\frac{\partial \mathbf{y^{T}}}{\partial \mathbf{W_2}}$')
    jacobian_transpose_test(resnet_layer.jac_dx_mul_v, resnet_layer.jac_transpose_dx_mul_v, n, n,
                            title=r'Jacobian Transpose Test Results - $\frac{\partial \mathbf{y^{T}}}{\partial \mathbf{x}}$')


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

    gradient_test(f, grad_f, x.size, x, "Gradient Test - Neural Network with ResNet Layer and Softmax Layer")

def section_2c():
    L = 5
    peaks_training_data, peak_validation_data = get_dataset("Peaks")
    X_train, C_train, n, l = peaks_training_data.X_raw, peaks_training_data.C, peaks_training_data.n, peaks_training_data.l

    nn = create_basic_neural_network_with_l_layers(l=L, input_dim=n, output_dim=l)
    gradient_test_for_nn(nn, X_train, C_train, "Gradient Test - Neural Network with L = 4 Layers And Softmax Layer")

    nn = create_neural_network_with_l_resnet_layers(l=L, input_dim=n, output_dim=l)
    gradient_test_for_nn(nn, X_train, C_train, "Gradient Test - Neural Network with L = 4 ResNet Layers And Softmax Layer")

    n_resnet = 4
    n_layers = 3

    nn = create_neural_network_with_mixed_layers(n_resnet=n_resnet, n_layers=n_layers, input_dim=n, output_dim=l)
    gradient_test_for_nn(nn, X_train, C_train, "Gradient Test - Neural Network with L = 4 ResNet Layers, K = 3 Layers And Softmax Layer")

def section_2d():
    for dataset in (["SwissRoll", "GMM", "Peaks"]):
        training_data, validation_data = get_dataset(dataset)
        X_raw, C, n, l = training_data.X_raw, training_data.C, training_data.n, training_data.l
        X_validation, C_validation = validation_data.X_raw, validation_data.C

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

        train_accuracy, validation_accuracy, loss_list = nn.train(X_train = X_raw,
                 C_train = C,
                 X_validation=X_validation,
                 C_validation=C_validation,
                 epochs=100,
                 mb_size = 16,
                 learning_rate = 0.01)

        plot_train_vs_validation_results(train_accuracy, validation_accuracy, x_label = "Epoch", y_label = "Accuracy",
                                         title=f"{dataset} - Accuracy vs. Epoch")

    for dataset in ["Peaks", "GMM", "SwissRoll"]:
        test_different_network_depths(dataset)
        test_different_learning_rates(dataset)
        test_different_batch_sizes(dataset)

    test_network_for_swissroll()

def section_2e():
    # for dataset in ["GMM", "Peaks", "SwissRoll"]:
    #     training_data, validation_data = get_dataset(dataset, train_percentage=0.008 if dataset != "SwissRoll" else 0.01)
    #     X_raw, C, n, l = training_data.X_raw, training_data.C, training_data.n, training_data.l
    #     X_validation, C_validation = validation_data.X_raw, validation_data.C
    #
    #     nn = NeuralNetwork(
    #         layers=[
    #             Layer(
    #                 input_dim=n,
    #                 output_dim=8,
    #                 activation=Activation.TANH
    #             ),
    #             Layer(
    #                 input_dim=8,
    #                 output_dim=16,
    #                 activation=Activation.TANH
    #             ),
    #             SoftmaxLayer(
    #                 input_dim=16,
    #                 output_dim=l
    #             )
    #         ]
    #     )
    #
    #     train_accuracy, validation_accuracy, loss_list = nn.train(X_train=X_raw,
    #                                                                   C_train=C,
    #                                                                   X_validation=X_validation,
    #                                                                   C_validation=C_validation,
    #                                                                   epochs=5000,
    #                                                                   mb_size=16,
    #                                                                   learning_rate=0.01)
    #     plot_train_vs_validation_results(train_accuracy, validation_accuracy, x_label = "Epoch", y_label = "Accuracy",
    #                                                  title=f"{dataset} - Accuracy vs. Epoch")

    # test_network_for_swissroll(train_percentage=0.01)
    get_train_percentage = lambda dataset: 0.01 if dataset == "SwissRoll" else 0.008
    for dataset in ["GMM", "SwissRoll", "Peaks"]:
        train_percentage = get_train_percentage(dataset)
        # test_different_network_depths(dataset, train_percentage)
        test_different_learning_rates(dataset, train_percentage)
        test_different_batch_sizes(dataset, train_percentage)

def section_1():
    section_1a()
    section_1b()
    section_1c()

def section_2():
    section_2a()
    section_2b()
    section_2c()
    section_2d()
    section_2e()

if __name__ == '__main__':
    section_2e()

