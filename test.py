import pickle

import numpy as np
from matplotlib import pyplot as plt

from data_utils import create_dataset_training_data
from figure_utils import plot_list
from gradient_descent import sgd
from models.dataset_training_data import DatasetTrainingData
from softmax_cross_entropy import softmax_cross_entropy_gradient, softmax_cross_entropy_loss, \
    softmax_cross_entropy_accuracy
from utils import flatten_weights_matrix_to_vector, reshape_weights_vector_to_matrix


def generate_mock_data_with_two_labels():
    m = 1000
    n = 2
    l = 2

    x = np.random.uniform(-10, 10, m)
    y = np.random.uniform(-10, 10, m)
    X = np.vstack((x, y))  # Stack x and y to form the feature matrix (2 x m)

    # Generate labels and C: matrix l x m (one-hot encoding for classes)
    labels = np.where(y >= x, 0, 1)  # 0 if above y = x, 1 if below
    C = np.zeros((l, m))  # Initialize one-hot matrix
    C[labels, np.arange(m)] = 1  # Fill one-hot encoding

    return X, C, labels

def plot_data_with_two_labels(X, labels):
    plt.figure(figsize=(8, 8))
    plt.scatter(X[0, labels == 0], X[1, labels == 0], c='blue', label='Label 0 (Above y = x)')
    plt.scatter(X[0, labels == 1], X[1, labels == 1], c='red', label='Label 1 (Below y = x)')
    plt.plot(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100), color='black', linestyle='--', linewidth=1,
             label='y = x')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Binary Classification: y = x')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_data_with_three_labels(X, labels):
    plt.figure(figsize=(8, 8))
    plt.scatter(X[0, labels == 0], X[1, labels == 0], c='blue', label='Label 0 (Above y = x)')
    plt.scatter(X[0, labels == 1], X[1, labels == 1], c='red', label='Label 1 (Below y = -x)')
    plt.scatter(X[0, labels == 2], X[1, labels == 2], c='green', label='Label 2 (Between y = -x and y = x)')
    plt.plot(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100), 'k--', label='y = x')  # y = x
    plt.plot(np.linspace(-10, 10, 100), -np.linspace(-10, 10, 100), 'k-.', label='y = -x')  # y = -x

    plt.xlabel('X[0] (Feature 1: x)')
    plt.ylabel('X[1] (Feature 2: y)')
    plt.title('Three-Class Classification: y = x and y = -x as Boundaries')
    plt.legend()
    plt.grid(True)
    plt.show()

def generate_mock_data_with_three_labels():
    m = 1000
    n = 2
    l = 3

    x = np.random.uniform(-10, 10, m)
    y = np.random.uniform(-10, 10, m)
    X = np.vstack((x, y))

    labels = np.zeros(m, dtype=int)
    labels[y < -x] = 1  # Label 1: below the line y = -x
    labels[(y >= -x) & (y <= x)] = 2  # Label 2: between y = -x and y = x

    C = np.zeros((l, m))
    C[labels, np.arange(m)] = 1

    return X, C, labels

def test_logistic_regression_with_two_labels():
    X, C, labels = generate_mock_data_with_two_labels()
    training_data = DatasetTrainingData(X, C)

    X = training_data.X
    C = training_data.C
    W = training_data.W

    initial_weights_vector = flatten_weights_matrix_to_vector(W)

    min_point, loss_progression_list, accuracy_list, _ = sgd(X, C, initial_weights_vector,
                                                                            grad_f=softmax_cross_entropy_gradient,
                                                                            loss_func=softmax_cross_entropy_loss,
                                                                            accuracy_func=softmax_cross_entropy_accuracy,
                                                                            batch_size=500,
                                                                            max_epochs=1000,
                                                                            is_samples_in_columns=True)
    plot_data_with_two_labels(X, labels)
    plot_list(values=loss_progression_list, x_label = "Iteration", y_label = "Loss", title="Loss progression", label="Loss progression")
    plot_list(values =accuracy_list, x_label = "Iteration", y_label = "Accuracy Percentage", title="Accuracy Progression", label="Accuracy Percentage")

def test_logistic_regression_with_three_labels():
    X, C, labels = generate_mock_data_with_three_labels()
    training_data = DatasetTrainingData(X, C)

    X = training_data.X
    C = training_data.C
    W = training_data.W

    initial_weights_vector = flatten_weights_matrix_to_vector(W)

    min_point, loss_progression_list, accuracy_list, _ = sgd(X, C, initial_weights_vector,
                                                                            grad_f=softmax_cross_entropy_gradient,
                                                                            loss_func=softmax_cross_entropy_loss,
                                                                            accuracy_func=softmax_cross_entropy_accuracy,
                                                                            batch_size=500,
                                                                            max_epochs=1000,
                                                                            is_samples_in_columns=True)
    plot_data_with_three_labels(X, labels)
    plot_list(values=loss_progression_list, x_label="Iteration", y_label="Loss", title="Loss progression",
              label="Loss progression")
    plot_list(values=accuracy_list, x_label="Iteration", y_label="Accuracy Percentage", title="Accuracy Progression",
              label="Accuracy Percentage")

def test_dataset_learning_rates_and_batch_sizes(dataset):
    dataset_training_data = create_dataset_training_data(dataset, 0.25)
    X, C, W = dataset_training_data.X, dataset_training_data.C, dataset_training_data.W
    initial_weights_vector = flatten_weights_matrix_to_vector(W)

    batch_sizes = [16, 32, 64, 128, 256, 512, 1024]
    learning_rates = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    results = {}

    for batch_size, learning_rate in zip(batch_sizes, learning_rates):
        min_point, loss_progression_list, accuracy_list, _ = sgd(X, C, initial_weights_vector,
                                                                 grad_f=softmax_cross_entropy_gradient,
                                                                 loss_func=softmax_cross_entropy_loss,
                                                                 accuracy_func=softmax_cross_entropy_accuracy,
                                                                 learning_rate=learning_rate,
                                                                 batch_size=batch_size,
                                                                 max_epochs=2500,
                                                                 is_samples_in_columns=True)
        key = f"ds_{dataset}_lr_{learning_rate}_bs_{batch_size}"

        results[key] = {
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "loss_progression": loss_progression_list,
            "accuracy_progression": accuracy_list,
            "min_point": min_point
        }

        plot_list(loss_progression_list, x_label="Iteration", y_label="Loss", label="Loss Progression",
                  title=f"{dataset}- Loss Progression During SGD - LR: {learning_rate}, batch_size: {batch_size}",
                  filename=f"output/ds_{dataset}_lr_{learning_rate}_bs_{batch_size}_loss")
        plot_list(accuracy_list, x_label="Iteration", y_label="Accuracy", label="Accuracy Progression",
                  title=f"{dataset} - Accuracy Progression During SGD - LR: {learning_rate}, batch_size: {batch_size}",
                  filename=f"output/ds_{dataset}_lr_{learning_rate}_bs_{batch_size}_accuracy")

    with open(f"output/{dataset}_sgd_results.pkl", "wb") as f:
        pickle.dump(results, f)


if __name__ == '__main__':
    test_logistic_regression_with_two_labels()
    test_logistic_regression_with_three_labels()
