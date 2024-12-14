import matplotlib.pyplot as plt
import numpy as np

def generate_gradient_test_plot(epsilons, first_equation, second_equation):
    plt.figure(figsize=(8, 6))

    plt.plot(epsilons, np.abs(first_equation), label=r'$|f(x + \epsilon d) - f(x)|$', marker='o')
    plt.plot(epsilons, np.abs(second_equation), label=r'$|f(x + \epsilon d) - f(x) - \epsilon d^T \nabla f_x|$', marker='x')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$\epsilon$', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.title('Gradient Test Results')
    plt.legend()
    plt.grid()
    plt.show()


def gradient_test(f, grad_f, dim):
    x = np.random.rand(dim)
    grad_f_x = grad_f(x)
    n_points = 15
    epsilons = 0.25 ** np.arange(n_points)
    d = np.random.randn(dim)
    d = d / np.linalg.norm(d)

    first_equation = np.array([np.abs(f(x+eps*d) - f(x)) for eps in epsilons])
    second_equation = np.array([np.abs(f(x+eps*d) - f(x) - eps*d.T@grad_f_x) for eps in epsilons])

    generate_gradient_test_plot(epsilons, first_equation, second_equation)

def validate_gradient_test():
    # Test on some simple examples
    gradient_test(lambda x: x ** 2, lambda x: 2 * x, 1)
    gradient_test(lambda v: np.sum(np.square(v)), lambda v: 2 * v, 5)

if __name__ == '__main__':
    validate_gradient_test()


