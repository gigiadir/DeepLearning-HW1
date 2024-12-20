import matplotlib.pyplot as plt
import numpy as np

from utils.figure_utils import plot_list


def generate_verification_test_plot(epsilons, first_equation, second_equation, title = 'Gradient Test Results'):
    plt.figure(figsize=(8, 6))

    plt.plot(range(1, len(epsilons)+1), np.abs(first_equation), label='Zero order approximation', marker='o')
    plt.plot(range(1, len(epsilons)+1), np.abs(second_equation), label='First order approximation', marker='x')
    plt.yscale('log')
    plt.xlabel(r'$\epsilon$', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()


'''
    f: R^{dim} -> R
    grad_f: R^{dim} -> R^{dim}
'''
def gradient_test(f, grad_f, dim, x = None):
    x = x if x is not None else np.random.rand(dim, 1)
    grad_f_x = grad_f(x)
    n_points = 15
    epsilons = 0.25 ** np.arange(n_points)
    d = np.random.randn(dim, 1)
    d = d / np.linalg.norm(d)

    first_equation = np.array([np.squeeze(np.abs(f(x+eps*d) - f(x))) for eps in epsilons])
    second_equation = np.array([np.squeeze(np.abs(f(x+eps*d) - f(x) - eps*d.T@grad_f_x)) for eps in epsilons])

    generate_verification_test_plot(epsilons, first_equation, second_equation)

def validate_gradient_test():
    # Test on some simple examples
   # gradient_test(lambda x: np.square(x), lambda x: 2 * x, 1)
   # gradient_test(lambda v: np.sum(np.square(v)), lambda v: 2 * v, 5)
    gradient_test(lambda x: x * 5, lambda _x: np.float16(5), 1)

def jacobian_test(f, jac_dx_v, dim):
    x = np.random.rand(dim, 1)
    n_points = 15
    epsilons = 0.25 ** np.arange(n_points)
    d = np.random.rand(dim, 1)
    d = d / np.linalg.norm(d)

    first_equation = np.array([np.linalg.norm(f(x+eps*d) - f(x), ord=2) for eps in epsilons])
    second_equation = np.array([np.linalg.norm(f(x+eps*d) - f(x) - jac_dx_v(x, eps*d), ord=2) for eps in epsilons])

    generate_verification_test_plot(epsilons, first_equation, second_equation, title="Jacobian Test Results")

def jacobian_transpose_test(jac_dx_v, jac_transpose_dx_v, dim_u, dim_v):
    num_iterations = 15
    errors = []

    for i in range(num_iterations):
        u = np.random.rand(dim_u, 1)
        v = np.random.rand(dim_v, 1)
        x = np.random.rand(dim_v, 1)
        first = u.T @ jac_dx_v(x, v)
        second = v.T @ jac_transpose_dx_v(x, u)
        error = np.abs(np.squeeze(first - second))
        errors.append(error)

    plot_list(errors, x_label = "Iteration", y_label = "Error", title="Transpose Test", label_train=None)


def validate_jacobian_test():
    def f(vec):
        x, y = vec[0], vec[1]
        f1 = x**2 + y**2
        f2 = x**2 - y**2

        return np.array([f1, f2])

    def jac_f_v(vec, v):
        x, y = vec[0], vec[1]
        v1, v2 = v[0], v[1]

        o1 = 2*x*v1 + 2*y*v2
        o2 = 2*x*v1 - 2*y*v2

        return np.array([o1, o2])

    def g(vec):
        u, v, x, y = vec[0], vec[1], vec[2], vec[3]
        g1 = u**2 * v * y
        g2 = x * y + u * v
        g3 = u * x + v**2 * y

        return np.array([g1, g2, g3])

    def jac_g_v(vec, v):
        u, w, x, y = vec[0], vec[1], vec[2], vec[3]
        v1, v2, v3, v4 = v[0], v[1], v[2], v[3]

        o1 = 2 * u * w * y * v1 + u**2 * y * v2 + u**2 * w * v4
        o2 = w * v1 + u * v2 + y * v3 + x * v4
        o3 = x * v1 + 2 * w * y * v2 + u * v3 + w**2 * v4

        return np.array([o1, o2, o3])

    jacobian_test(f, jac_f_v, 2)
    jacobian_test(g, jac_g_v, 4)



if __name__ == '__main__':
    validate_gradient_test()
    #validate_jacobian_test()


