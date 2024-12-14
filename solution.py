import numpy as np

from data_utils import create_dataset_training_data
from gradient_descent import sgd
from gradient_test import gradient_test
from least_squares import generate_least_squares, least_squares_gradient, plot_gradient_descent_least_squares_result
from softmax_cross_entropy import softmax_cross_entropy, softmax_cross_entropy_gradient


def section_1a():
    peaks_training_data = create_dataset_training_data("Peaks")
    X = peaks_training_data.X
    C = peaks_training_data.C
    n = peaks_training_data.n
    l = peaks_training_data.l

    softmax_cross_entropy_func = lambda w : softmax_cross_entropy(X, C, w)
    softmax_cross_entropy_gradient_func = lambda w : softmax_cross_entropy_gradient(X, C, w)

    gradient_test(softmax_cross_entropy_func, softmax_cross_entropy_gradient_func, (n+1) * l)

def section_1b():
    m = 20
    A, b, x_points = generate_least_squares(m)
    x = np.random.rand(2, 1)
    min_point, progression_list = sgd(A, b, x, least_squares_gradient, 5, max_iter=100, tolerance=-0.01)
    for i in range(0, len(progression_list), 10):
        plot_gradient_descent_least_squares_result(x_points, b, progression_list[i])

def section_1c():
    pass

def main():
    section_1a()
    section_1b()
    section_1c()



if __name__ == '__main__':
    main()