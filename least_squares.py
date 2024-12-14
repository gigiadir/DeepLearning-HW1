import numpy as np
import matplotlib.pyplot as plt

from gradient_test import gradient_test


'''
'''
def least_squares_loss(A, b, x):
    loss = 0.5 * np.sum(np.square(A@x - b))

    return loss, None


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


def plot_gradient_descent_least_squares_result(x, y, least_squares_result):
    # plot the results
    plt.figure(figsize=(10, 8))
    plt.plot(x, y, 'b.')
    plt.plot(x, least_squares_result[0] * x + least_squares_result[1], 'r')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


