import numpy as np
import matplotlib.pyplot as plt

from tests.gradient_and_jacobian_test import gradient_test


def least_squares_loss(A, b, x):
    loss = 0.5 * np.sum(np.square(A@x - b))

    return float(loss)


def least_squares_gradient(A, b, x):
    gradient_vector = A.T@(A@x - b)

    return gradient_vector


'''
creates A : m x 1
        b : m x 1
        with m >= n >= 1
        where the expected result is 2*x
'''
def generate_least_squares(m):
    x = np.linspace(0, 1, m)
    b = 1 + x + x * np.random.random(m)
    #b = 1 + x
    b = b.reshape(m,1)

    A = np.vstack([x, np.ones(m)]).T

    return A, b, x

def validate_least_squares_gradient(A, b):
    least_squares_loss_func = lambda x : least_squares_loss(A, b, x)
    least_squares_gradient_func = lambda x : least_squares_gradient(A, b, x)

    gradient_test(least_squares_loss_func, least_squares_gradient_func, 2)


def plot_line_fitting(A, b, theta_list, x_points, method, interval=50, final_theta = None):
    plt.scatter(x_points, b, label="Data points", color="blue", alpha=0.5)
    final_theta =  final_theta if final_theta is not None else theta_list[-1]
    for i, theta in enumerate(theta_list[::interval]):
        y_fit = A @ theta
        plt.plot(x_points, y_fit, label=f"Iteration {i * interval}", alpha=0.7)
    plt.plot(x_points, A @ final_theta, label="Final fit", color="red", linewidth=2)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"{method} - Least Squares Line Fitting Progress")
    plt.legend()
    plt.savefig(f"output/Section 1b/{method}-least-squares.png")
    plt.show()

