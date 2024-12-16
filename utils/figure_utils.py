from matplotlib import pyplot as plt


def plot_list(values_train, values_validation = None, x_label = "Iteration", y_label = "Value", title = "Plot",
              label_train = "Training", label_validation = "Validation", filename = ''):
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(values_train) + 1), values_train, label=label_train, color='b')

    if values_validation:
        plt.plot(range(1, len(values_validation) + 1), values_validation, label=label_validation, color='r', linestyle='dashed')

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid()
    plt.legend()
    plt.savefig(f"{filename or title}.png", dpi=300, bbox_inches='tight')
    plt.show()
