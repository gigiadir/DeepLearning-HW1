from matplotlib import pyplot as plt


def plot_list(values, x_label, y_label, title, label, filename = ''):
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(values) + 1), values, label=label, color='b')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid()
    plt.legend()
    plt.savefig(f"{filename or label}.png", dpi=300, bbox_inches='tight')
    plt.show()
