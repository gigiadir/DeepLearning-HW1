import pickle
from itertools import product

import numpy as np
from matplotlib import pyplot as plt

from utils.data_utils import get_dataset
from utils.figure_utils import plot_list
from gradient_descent import sgd
from models.dataset_training_data import DatasetTrainingData
from softmax_cross_entropy import softmax_cross_entropy_gradient, softmax_cross_entropy_loss, \
    softmax_cross_entropy_accuracy
from utils.vector_utils import flatten_weights_matrix_to_vector


def generate_mock_data_with_two_labels():
    m = 1000
    n = 2
    l = 2

    x = np.random.uniform(-10, 10, m)
    y = np.random.uniform(-10, 10, m)
    X = np.vstack((x, y))

    labels = np.where(y >= x, 0, 1)
    C = np.zeros((l, m))
    C[labels, np.arange(m)] = 1

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
    plt.plot(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100), 'k--', label='y = x')
    plt.plot(np.linspace(-10, 10, 100), -np.linspace(-10, 10, 100), 'k-.', label='y = -x')

    plt.xlabel('x')
    plt.ylabel('y')
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

    X, C, W = training_data.X, training_data.C, training_data.W
    initial_weights_vector = flatten_weights_matrix_to_vector(W)

    min_point, loss_progression_list, accuracy_list, _ = sgd(X, C, initial_weights_vector,
                                                                            grad_f=softmax_cross_entropy_gradient,
                                                                            loss_func=softmax_cross_entropy_loss,
                                                                            accuracy_func=softmax_cross_entropy_accuracy,
                                                                            batch_size=32,
                                                                            max_epochs=1000,
                                                                            is_samples_in_columns=True)
    plot_data_with_two_labels(X, labels)
    plot_list(values_train=loss_progression_list, x_label = "Iteration", y_label = "Loss", title="Loss progression")
    plot_list(values_train=accuracy_list, x_label = "Iteration", y_label = "Accuracy Percentage", title="Accuracy Progression")

def test_logistic_regression_with_three_labels():
    X, C, labels = generate_mock_data_with_three_labels()
    training_data = DatasetTrainingData(X, C)

    X, C, W = training_data.X, training_data.C, training_data.W
    initial_weights_vector = flatten_weights_matrix_to_vector(W)

    min_point, loss_progression_list, accuracy_list, _ = sgd(X, C, initial_weights_vector,
                                                                            grad_f=softmax_cross_entropy_gradient,
                                                                            loss_func=softmax_cross_entropy_loss,
                                                                            accuracy_func=softmax_cross_entropy_accuracy,
                                                                            batch_size=500,
                                                                            max_epochs=1000,
                                                                            is_samples_in_columns=True)
    plot_data_with_three_labels(X, labels)
    plot_list(values_train=loss_progression_list, x_label="Iteration", y_label="Loss", title="Loss progression")
    plot_list(values_train=accuracy_list, x_label="Iteration", y_label="Accuracy Percentage", title="Accuracy Progression")

def test_dataset_learning_rates_and_batch_sizes(dataset):
    dataset_training_data, dataset_validation_data = get_dataset(dataset, 0.25)
    X_train, C_train, W = dataset_training_data.X, dataset_training_data.C, dataset_training_data.W
    X_validation, C_validation = dataset_validation_data.X, dataset_validation_data.C

    initial_weights_vector = flatten_weights_matrix_to_vector(W)

    batch_sizes = [16, 32, 64, 128, 256, 512, 1024]
    learning_rates = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    results = {}

    for batch_size, learning_rate in product(batch_sizes, learning_rates):
        print(f"Testing dataset {dataset}, learning rate={learning_rate}, batch size={batch_size}")
        min_point, loss_progression_list, train_accuracy_list, weights_list = sgd(X_train, C_train, initial_weights_vector,
                                                                 grad_f=softmax_cross_entropy_gradient,
                                                                 loss_func=softmax_cross_entropy_loss,
                                                                 accuracy_func=softmax_cross_entropy_accuracy,
                                                                 learning_rate=learning_rate,
                                                                 batch_size=batch_size,
                                                                 max_epochs=5000,
                                                                 is_samples_in_columns=True)

        validation_accuracy_list = [softmax_cross_entropy_accuracy(X_validation, C_validation, w) for w in weights_list]
        key = f"ds_{dataset}_lr_{learning_rate}_bs_{batch_size}"

        results[key] = {
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "loss_progression": loss_progression_list,
            "accuracy_progression": train_accuracy_list,
            "validation_accuracy": validation_accuracy_list,
            "min_point": min_point
        }

        plot_list(loss_progression_list, x_label="Iteration", y_label="Loss",
                  title=f"{dataset}- Loss Progression During SGD - LR: {learning_rate}, batch_size: {batch_size}",
                  filename=f"output/ds_{dataset}_lr_{learning_rate}_bs_{batch_size}_loss")
        plot_list(train_accuracy_list, validation_accuracy_list, x_label="Iteration", y_label="Accuracy",
                  title=f"{dataset} - Accuracy Progression During SGD - LR: {learning_rate}, batch_size: {batch_size}",
                  filename=f"output/ds_{dataset}_lr_{learning_rate}_bs_{batch_size}_accuracy")

    with open(f"output/{dataset}_sgd_results.pkl", "wb") as f:
        pickle.dump(results, f)


if __name__ == '__main__':
    test_logistic_regression_with_two_labels()
    test_logistic_regression_with_three_labels()