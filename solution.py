import numpy as np

from utils.data_utils import get_dataset
from gradient_descent import sgd
from gradient_and_jacobian_test import gradient_test
from least_squares import generate_least_squares, least_squares_gradient, plot_gradient_descent_least_squares_result, \
    least_squares_loss
from softmax_cross_entropy import softmax_cross_entropy_loss, softmax_cross_entropy_gradient
from sgd_test import test_dataset_learning_rates_and_batch_sizes


def section_1a():
    peaks_training_data, _ = get_dataset("Peaks")
    X, C, n, l = peaks_training_data.X, peaks_training_data.C, peaks_training_data.n, peaks_training_data.l

    softmax_cross_entropy_func = lambda w : softmax_cross_entropy_loss(X, C, w)
    softmax_cross_entropy_gradient_func = lambda w : softmax_cross_entropy_gradient(X, C, w)

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

def main():
    section_1a()
    section_1b()
    section_1c()



if __name__ == '__main__':
    main()