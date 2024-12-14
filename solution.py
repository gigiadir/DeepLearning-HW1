import numpy as np
import pickle

from data_utils import create_dataset_training_data
from figure_utils import plot_list
from gradient_descent import sgd
from gradient_test import gradient_test
from least_squares import generate_least_squares, least_squares_gradient, plot_gradient_descent_least_squares_result, \
    least_squares_loss
from softmax_cross_entropy import softmax_cross_entropy_loss, softmax_cross_entropy_gradient, \
    softmax_cross_entropy_accuracy
from utils import flatten_weights_matrix_to_vector


def section_1a():
    peaks_training_data = create_dataset_training_data("Peaks")
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
                                              max_epochs=100,
                                              tolerance=-0.01)
    for i in range(0, len(theta_list), 30):
        plot_gradient_descent_least_squares_result(x_points, b, theta_list[i])


def section_1c():
    peaks_training_data = create_dataset_training_data("Peaks", 0.25)
    X, C, W = peaks_training_data.X, peaks_training_data.C, peaks_training_data.W
    initial_weights_vector = flatten_weights_matrix_to_vector(W)

    batch_sizes = [16, 32, 64, 128, 256, 512, 1024]
    learning_rates = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    results = {}
    batch_sizes = [16]
    learning_rates = [0.01]

    for batch_size, learning_rate in zip(batch_sizes, learning_rates):
        min_point, loss_progression_list, accuracy_list, _ = sgd(X, C, initial_weights_vector,
                                                                    grad_f=softmax_cross_entropy_gradient,
                                                                    loss_func=softmax_cross_entropy_loss,
                                                                    accuracy_func=softmax_cross_entropy_accuracy,
                                                                    learning_rate=learning_rate,
                                                                    batch_size=batch_size,
                                                                    max_epochs=200,
                                                                    is_samples_in_columns=True)
        key = f"lr_{learning_rate}_bs_{batch_size}"
        results[key] = {
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "loss_progression": loss_progression_list,
            "accuracy_progression": accuracy_list,
            "min_point": min_point
        }

        plot_list(loss_progression_list, x_label="Iteration", y_label="Loss", label="Loss Progression", title=f"Peaks- Loss Progression During SGD - LR: {learning_rate}, batch_size: {batch_size}", filename=f"output/lr_{learning_rate}_bs_{batch_size}_loss")
        plot_list(accuracy_list, x_label="Iteration", y_label="Accuracy", label="Accuracy Progression", title=f"Peaks - Accuracy Progression During SGD - LR: {learning_rate}, batch_size: {batch_size}", filename=f"output/lr_{learning_rate}_bs_{batch_size}_accuracy")

    with open("output/sgd_results.pkl", "wb") as f:
        pickle.dump(results, f)


def main():
    #section_1a()
    #section_1b()
    section_1c()



if __name__ == '__main__':
    main()