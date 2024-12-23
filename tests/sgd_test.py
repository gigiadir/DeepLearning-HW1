import numpy as np
from matplotlib import pyplot as plt

from utils.data_utils import get_dataset
from utils.figure_utils import plot_train_vs_validation_results
from gradient_descent import sgd, sgd_with_momentum
from models.dataset_training_data import DatasetTrainingData
from softmax_cross_entropy import softmax_cross_entropy_gradient_dw, softmax_cross_entropy_loss, \
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
    plt.savefig(f"output/Section 1b/Two Labels Data Plot.png", dpi=300, bbox_inches='tight')
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

    min_point, loss_progression_list, accuracy_list, _ = sgd_with_momentum(X, C, initial_weights_vector,
                                                                           grad_f=softmax_cross_entropy_gradient_dw,
                                                                           loss_func=softmax_cross_entropy_loss,
                                                                           accuracy_func=softmax_cross_entropy_accuracy,
                                                                           batch_size=32,
                                                                           max_epochs=300,
                                                                           is_samples_in_columns=True)
    plot_data_with_two_labels(X, labels)
    plot_train_vs_validation_results(values_train=loss_progression_list, x_label ="Iteration", y_label ="Loss", title="Loss progression")
    plot_train_vs_validation_results(values_train=accuracy_list, x_label ="Iteration", y_label ="Accuracy Percentage", title="Accuracy Progression")

def test_logistic_regression_with_three_labels():
    X, C, labels = generate_mock_data_with_three_labels()
    training_data = DatasetTrainingData(X, C)

    X, C, W = training_data.X, training_data.C, training_data.W
    initial_weights_vector = flatten_weights_matrix_to_vector(W)

    min_point, loss_progression_list, accuracy_list, _ = sgd_with_momentum(X, C, initial_weights_vector,
                                                                           grad_f=softmax_cross_entropy_gradient_dw,
                                                                           loss_func=softmax_cross_entropy_loss,
                                                                           accuracy_func=softmax_cross_entropy_accuracy,
                                                                           batch_size=32,
                                                                           max_epochs=300,
                                                                           is_samples_in_columns=True)
    plot_data_with_three_labels(X, labels)
    plot_train_vs_validation_results(values_train=loss_progression_list, x_label="Iteration", y_label="Loss", title="Loss progression")
    plot_train_vs_validation_results(values_train=accuracy_list, x_label="Iteration", y_label="Accuracy Percentage", title="Accuracy Progression")

def compare_different_batch_sizes(dataset):
    # Sampled 10% of 20K-25K points datasets
    dataset_training_data, dataset_validation_data = get_dataset(dataset, 0.25)
    X_train, C_train, W = dataset_training_data.X, dataset_training_data.C, dataset_training_data.W
    X_validation, C_validation = dataset_validation_data.X, dataset_validation_data.C

    initial_weights_vector = flatten_weights_matrix_to_vector(W)

    batch_sizes = [16, 64, 512, 1024]
    train_accuracy_per_batch_size = {}
    validation_accuracy_per_batch_size = {}
    max_epochs = 150
    for batch_size in batch_sizes:
        min_point, loss_progression_list, train_accuracy_list, theta_list = sgd_with_momentum(
            X_train, C_train, initial_weights_vector,
            grad_f=softmax_cross_entropy_gradient_dw,
            loss_func=softmax_cross_entropy_loss,
            accuracy_func=softmax_cross_entropy_accuracy,
            batch_size=batch_size,
            max_epochs=max_epochs,
            learning_rate=0.01,
            is_samples_in_columns=True
        )

        train_accuracy_per_batch_size[batch_size] = train_accuracy_list
        validation_accuracy_per_batch_size[batch_size] = [softmax_cross_entropy_accuracy(X_validation, C_validation, theta) for theta in theta_list]

    plt.figure(figsize=(10, 6))
    for batch_size in batch_sizes:
        plt.plot(train_accuracy_per_batch_size[batch_size], label=f"Batch Size {batch_size}")

    plt.xlabel("Epoch")
    plt.ylabel("Train Set Accuracy")
    plt.title(f"{dataset} - Softmax Accuracy On Training Set With SGD With Momentum")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"output/Section 1c/{dataset} - Softmax Accuracy Training With SGD With momentum By Batch Size - {max_epochs} iterations - Start {batch_sizes[0]}")
    plt.show()

    plt.figure(figsize=(10, 6))
    for batch_size in batch_sizes:
        plt.plot(validation_accuracy_per_batch_size[batch_size], label=f"Batch Size {batch_size}")

    plt.xlabel("Epoch")
    plt.ylabel("Validation Set Accuracy")
    plt.title(f"{dataset} - Softmax Accuracy On Validation Set With SGD With Momentum")
    plt.legend()
    plt.grid(True)
    plt.savefig(
        f"output/Section 1c/{dataset} - Softmax Accuracy Validation With SGD With momentum By Batch Size - {max_epochs} iterations - Start {batch_sizes[0]}")
    plt.show()


def compare_different_learning_rates(dataset):
    dataset_training_data, dataset_validation_data = get_dataset(dataset, 0.25)
    X_train, C_train, W = dataset_training_data.X, dataset_training_data.C, dataset_training_data.W
    X_validation, C_validation = dataset_validation_data.X, dataset_validation_data.C

    initial_weights_vector = flatten_weights_matrix_to_vector(W)

    learning_rates = [0.001, 0.01, 0.1]
    train_accuracy_per_learning_rate = {}
    validation_accuracy_per_learning_rate = {}

    for learning_rate in learning_rates:
        min_point, loss_progression_list, train_accuracy_list, theta_list = sgd_with_momentum(
            X_train, C_train, initial_weights_vector,
            grad_f=softmax_cross_entropy_gradient_dw,
            loss_func=softmax_cross_entropy_loss,
            accuracy_func=softmax_cross_entropy_accuracy,
            batch_size=64,
            max_epochs=200,
            learning_rate=learning_rate,
            is_samples_in_columns=True
        )

        train_accuracy_per_learning_rate[learning_rate] = train_accuracy_list
        validation_accuracy_per_learning_rate[learning_rate] = [
            softmax_cross_entropy_accuracy(X_validation, C_validation, theta) for theta in theta_list]

    plt.figure(figsize=(10, 6))
    for learning_rate in learning_rates:
        plt.plot(train_accuracy_per_learning_rate[learning_rate], label=f"{learning_rate}")

    plt.xlabel("Epoch")
    plt.ylabel("Train Set Accuracy")
    plt.title(f"{dataset} - Softmax Accuracy On Training Set With SGD With Momentum")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"output/Section 1c/{dataset} - Softmax Accuracy Training With SGD With momentum By Learning Rate")
    plt.show()

    plt.figure(figsize=(10, 6))
    for learning_rate in learning_rates:
        plt.plot(validation_accuracy_per_learning_rate[learning_rate], label=f"{learning_rate}")

    plt.xlabel("Epoch")
    plt.ylabel("Validation Set Accuracy")
    plt.title(f"{dataset} - Softmax Accuracy On Validation Set With SGD With Momentum")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"output/Section 1c/{dataset} - Softmax Accuracy Validation With SGD With momentum By Learning Rate")
    plt.show()


if __name__ == '__main__':
    test_logistic_regression_with_two_labels()
    test_logistic_regression_with_three_labels()
