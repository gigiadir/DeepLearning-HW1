from matplotlib import pyplot as plt

from activation import Activation
from utils.data_utils import get_dataset
from utils.neural_network_utils import create_basic_neural_network_with_l_layers


def test_different_network_depths(dataset):
    layers_nums = [1, 3, 5, 10, 20]
    dataset_training_data, dataset_validation_data = get_dataset(dataset)
    X_train, C_train, n, l = dataset_training_data.X_raw, dataset_training_data.C, dataset_training_data.n, dataset_training_data.l
    X_validation, C_validation = dataset_validation_data.X_raw, dataset_validation_data.C

    validation_accuracies_by_layer = {}

    for layer_num in layers_nums:
        nn = create_basic_neural_network_with_l_layers(layer_num, n, l)
        train_accuracy, validation_accuracy, _ = nn.train(X_train, C_train, X_validation,
                                                          C_validation,
                                                          epochs = 100,
                                                          mb_size = 32,
                                                          learning_rate = 0.01)
        validation_accuracies_by_layer[layer_num] = validation_accuracy

    plt.figure(figsize=(10, 6))
    for layer_num, val_acc in validation_accuracies_by_layer.items():
        plt.plot(range(1, len(val_acc) + 1), val_acc, label=f"{layer_num} Layers")

    plt.title(f"{dataset} - Validation Accuracy by Epoch for Different Depths")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.legend(title="Validation Accuracy by Epoch for Different Depths")
    plt.grid(True)
    plt.savefig(f"output/Section 2d/{dataset} - accuracy by depth.png")
    plt.show()



def test_different_learning_rates(dataset):
    learning_rates = [0.001, 0.01, 0.1]
    dataset_training_data, dataset_validation_data = get_dataset(dataset)
    X_train, C_train, n, l = dataset_training_data.X_raw, dataset_training_data.C, dataset_training_data.n, dataset_training_data.l
    X_validation, C_validation = dataset_validation_data.X_raw, dataset_validation_data.C

    validation_accuracies_by_learning_rate = {}

    for learning_rate in learning_rates:
        nn = create_basic_neural_network_with_l_layers(8, n, l)
        train_accuracy, validation_accuracy, _ = nn.train(X_train, C_train, X_validation,
                                                          C_validation,
                                                          epochs = 100,
                                                          mb_size = 64,
                                                          learning_rate = learning_rate)
        validation_accuracies_by_learning_rate[learning_rate] = validation_accuracy

    plt.figure(figsize=(10, 6))
    for learning_rate, val_acc in validation_accuracies_by_learning_rate.items():
        plt.plot(range(1, len(val_acc) + 1), val_acc, label=f"{learning_rate}")

    plt.title(f"{dataset} - Validation Accuracy by Epoch for Different Learning Rates")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.legend(title="Validation Accuracy by Epoch for Different Learning Rates")
    plt.grid(True)
    plt.savefig(f"output/Section 2d/{dataset} - accuracy by learning rate.png")
    plt.show()

def test_different_batch_sizes(dataset):
    batch_sizes = [16, 64, 256, 1024]
    dataset_training_data, dataset_validation_data = get_dataset(dataset)
    X_train, C_train, n, l = dataset_training_data.X_raw, dataset_training_data.C, dataset_training_data.n, dataset_training_data.l
    X_validation, C_validation = dataset_validation_data.X_raw, dataset_validation_data.C

    validation_accuracies_by_batch_size = {}

    for batch_size in batch_sizes:
        nn = create_basic_neural_network_with_l_layers(8, n, l)
        train_accuracy, validation_accuracy, _ = nn.train(X_train, C_train, X_validation,
                                                          C_validation,
                                                          epochs = 100,
                                                          mb_size = batch_size,
                                                          learning_rate = 0.01)
        validation_accuracies_by_batch_size[batch_size] = validation_accuracy

    plt.figure(figsize=(10, 6))
    for batch_size, val_acc in validation_accuracies_by_batch_size.items():
        plt.plot(range(1, len(val_acc) + 1), val_acc, label=f"{batch_size}")

    plt.title(f"{dataset} - Validation Accuracy by Epoch for Different Batch Sizes")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.legend(title="Validation Accuracy by Epoch for Different Batch Sizes")
    plt.grid(True)
    plt.savefig(f"output/Section 2d/{dataset} - accuracy by batch size.png")
    plt.show()



