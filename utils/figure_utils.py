from matplotlib import pyplot as plt


def plot_train_vs_validation_results(values_train, values_validation = None, x_label ="Iteration", y_label ="Value", title ="Plot",
                                     label_train = "Training", label_validation = "Validation", filename = None):
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(values_train) + 1), values_train, label=label_train, color='b')

    if values_validation:
        plt.plot(range(1, len(values_train) + 1), values_validation, label=label_validation, color='r', linestyle='dashed')

    # plt.ylim(0, 0.1)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid()
    plt.legend()
    if filename is not None:
        plt.savefig(f"{filename}.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_loss_over_iterations(loss_list, method_name):
    plt.plot(loss_list, label=method_name)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Loss Progress")
    plt.legend()