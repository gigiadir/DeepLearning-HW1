import numpy as np
from matplotlib import pyplot as plt

from data_utils import create_dataset_training_data
from gradient_descent import sgd
from gradient_test import gradient_test
from least_squares import generate_least_squares, least_squares_gradient, plot_gradient_descent_least_squares_result, \
    least_squares_loss
from softmax_cross_entropy import softmax_cross_entropy, softmax_cross_entropy_gradient
from utils import flatten_weights_matrix_to_vector


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
    min_point, theta_list, loss_list = sgd(A, b, x, least_squares_gradient, least_squares_loss, 5, max_iter=100, tolerance=-0.01)
    for i in range(0, len(theta_list), 30):
        plot_gradient_descent_least_squares_result(x_points, b, theta_list[i])


def section_1c():
    peaks_training_data = create_dataset_training_data("Peaks")
    X = peaks_training_data.X
    C = peaks_training_data.C
    W = peaks_training_data.W

    initial_weights_vector = flatten_weights_matrix_to_vector(W)

    min_point, progression_list, loss_progression_list = sgd(X, C, initial_weights_vector, softmax_cross_entropy_gradient,
                                                             softmax_cross_entropy, batch_size=100, max_iter=1000, is_samples_in_columns=True)

    plt.figure(figsize=(8, 6))  # Optional: Set the figure size
    plt.plot(range(1, len(loss_progression_list) + 1), loss_progression_list, label="Loss Progression", color='b')
    plt.xlabel("Iteration")  # X-axis label
    plt.ylabel("Loss Value")  # Y-axis label
    plt.title("Loss Progression During Training")  # Plot title
    plt.grid()  # Add grid
    plt.legend()  # Add legend
    plt.show()  # Display the plot

def main():
    section_1a()
    section_1b()
    section_1c()



if __name__ == '__main__':
    main()