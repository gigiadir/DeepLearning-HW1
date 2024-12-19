import numpy as np

from layers.softmax_layer import SoftmaxLayer
from tests.gradient_and_jacobian_test import generate_verification_test_plot


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

    f = lambda X : softmax_layer.forward(X, C_random)

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
    grad_f_w = softmax_layer.get_output_grad()

    f = lambda w_vector: softmax_layer.forward(X=X, C=C_random, w_vector=w_vector)


    n_points = 10
    epsilons = 0.25 ** np.arange(n_points)
    d = np.random.randn(n+1, l)
    d = d / np.linalg.norm(d)
    d = d.flatten(order='F').reshape((n+1) * l, 1)
    w_vector = softmax_layer.w_vector

    first_equation = np.array([np.abs(f(w_vector + eps * d) - f(w_vector)) for eps in epsilons])

    second_equation = np.array([np.abs(f(w_vector + eps * d) - f(w_vector) - eps * d.T @ grad_f_w) for eps in epsilons])

    generate_verification_test_plot(epsilons, first_equation, second_equation)

if __name__ == '__main__':
    test_softmax_layer_gradient_dx()
    test_softmax_layer_gradient_dw()