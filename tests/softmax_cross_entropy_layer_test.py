import numpy as np

from layers.softmax_layer import SoftmaxLayer
from tests.gradient_and_jacobian_test import generate_verification_test_plot, gradient_test


def test_softmax_layer_gradient_dx():
    n = 2
    l = 5
    mb_size = 10

    softmax_layer = SoftmaxLayer(
        input_dim= n,
        output_dim= l
    )

    X = np.random.rand(n, mb_size)
    class_indices = np.random.randint(0, l, size=mb_size)
    C_random = np.eye(l)[:, class_indices]


    softmax_layer.forward(X, C_random)
    grad_f_x = softmax_layer.backprop(None)

    f = lambda X : softmax_layer.forward(X, C_random)[1]

    n_points = 10
    epsilons = 0.25 ** np.arange(n_points)
    d = np.random.randn(n, mb_size)
    d = d / np.linalg.norm(d)

    first_equation = np.array([np.abs(f(X+eps*d) - f(X)) for eps in epsilons])
    d_vector = d.flatten(order='F')
    second_equation = np.array([np.abs(f(X+eps*d) - f(X) - eps*d_vector.T@grad_f_x) for eps in epsilons])

    generate_verification_test_plot(epsilons, first_equation, second_equation)


def test_softmax_layer_gradient_dw():
    n = 2
    l = 5
    mb_size = 10

    softmax_layer = SoftmaxLayer(
        input_dim=n,
        output_dim=l
    )

    X = np.random.rand(n, mb_size)
    class_indices = np.random.randint(0, l, size=mb_size)
    C_random = np.eye(l)[:, class_indices]

    softmax_layer.forward(X, C_random)
    softmax_layer.backprop(None)

    f = lambda w_vector: softmax_layer.get_loss_with_specific_weights(w_vector)
    grad_f = lambda w_vector : softmax_layer.get_gradient_dw_at_weights(w_vector)

    gradient_test(f, grad_f, (n+1)*l)


if __name__ == '__main__':
    test_softmax_layer_gradient_dx()
    test_softmax_layer_gradient_dw()