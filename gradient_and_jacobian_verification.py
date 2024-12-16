import matplotlib.pyplot as plt
import numpy as np

def generate_verification_test_plot(epsilons, first_equation, second_equation,
                                    first_equation_label=r'$|f(x + \epsilon d) - f(x)|$',
                                    second_equation_label=r'$|f(x + \epsilon d) - f(x) - \epsilon d^T \nabla f_x|$'):
    plt.figure(figsize=(8, 6))

    plt.plot(range(1, len(epsilons) + 1), np.abs(first_equation), label=first_equation_label, marker='o')
    plt.plot(range(1, len(epsilons) + 1), np.abs(second_equation), label=second_equation_label, marker='x')
    plt.yscale('log')
    plt.xlabel('k', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.title('Gradient Test Results')
    plt.legend()
    plt.grid()
    plt.show()


'''
    f: R^{dim} -> R
    grad_f: R^{dim} -> R^{dim}
'''
def gradient_test(f, grad_f, dim):
    x = np.random.rand(dim)
    grad_f_x = grad_f(x)
    n_points = 10
    epsilons = 0.25 ** np.arange(1, n_points)
    d = np.random.randn(dim)
    d = d / np.linalg.norm(d)

    first_equation = np.array([np.abs(f(x+eps*d) - f(x)) for eps in epsilons])
    second_equation = np.array([np.abs(f(x+eps*d) - f(x) - eps*d.T@grad_f_x) for eps in epsilons])

    generate_verification_test_plot(epsilons, first_equation, second_equation)

def validate_gradient_test():
    # Test on some simple examples
    gradient_test(lambda x: x ** 2, lambda x: 2 * x, 1)
    gradient_test(lambda v: np.sum(np.square(v)), lambda v: 2 * v, 5)


def jacobian_test(f, jac_f, dim):
    x = np.random.randn(dim)
    n_points = 15
    epsilons = 0.25 ** np.arange(n_points)
    d = np.random.randn(dim)
    d = d / np.linalg.norm(d)

    first_equation = np.array([np.abs(f(x + eps * d) - f(x)) for eps in epsilons])
    second_equation = []
    generate_verification_test_plot(epsilons, first_equation, second_equation, second_equation_label=r'$|f(x + \epsilon d) - f(x) - JacMV(x, \epsilon d)|$')


if __name__ == '__main__':
    validate_gradient_test()


