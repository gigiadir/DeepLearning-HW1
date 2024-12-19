import numpy as np

from activation import Activation, get_activation, get_activation_derivative


class ResNetLayer:

    def __init__(self, dim, activation: Activation):
        self.W1 = np.random.randn(dim, dim)
        self.W2 = np.random.randn(dim, dim)
        self.b = np.random.randn(dim, 1)
        self.activation = get_activation(activation)
        self.activation_derivative = get_activation_derivative(activation)
        self.dim = dim

        self.X = None
        self.grad_b = None
        self.grad_W1 = None
        self.grad_W2 = None
        self.mb_size = None

    def forward(self, X, C):
        self.X = X
        self.mb_size = X.shape[1]

        linear_part = self.W1 @ X + self.b
        result = X + self.W2 @ self.activation(linear_part)

        return result, C


    def backward(self, next_grad_dx):
        grad_x = self.jac_transpose_dx_mul_v(self.X, next_grad_dx)

        self.grad_W1 = self.jac_transpose_dw1_mul_v(self.W1, next_grad_dx)
        self.grad_W2 = self.jac_transpose_dw2_mul_v(self.W2, next_grad_dx)
        self.grad_b = self.jac_transpose_db_mul_v(self.b, next_grad_dx)

        return grad_x

    def jac_dw1_mul_v(self, W1, v_vector):
        linear_part = W1 @ self.X + self.b
        result = self.W2 @ (self.activation_derivative(linear_part) * (v_vector @ self.X))

        return result

    def jac_transpose_dw1_mul_v(self, W1, v_vector):
        linear_part = W1 @ self.X + self.b
        result = (linear_part * (self.W2.T @ v_vector)) @ self.X.T

        return result

    def jac_dw2_mul_v(self, _W2, v_vector):
        linear_part = self.W1 @ self.X + self.b
        result = v_vector @ self.activation(linear_part)

        return result

    def jac_transpose_dw2_mul_v(self, _W2, v_vector):
        linear_part = self.W1 @ self.X + self.b
        result = v_vector @ self.activation(linear_part).T

        return result


    def jac_db_mul_v(self, b, v_vector):
        V = v_vector # reshape to wanted form
        linear_part = self.W1 @ self.X + b
        result = self.W2 @ (self.activation(linear_part) * V)

        return result

    def jac_transpose_db_mul_v(self, b, v_vector):
        V = v_vector # reshape to wanted form
        linear_part = self.W1 @ self.X + b
        result = self.activation(linear_part) * (self.W2 * V)

        return result

    def jac_dx_mul_v(self, X, v_vector):
        V = v_vector # reshape to wanted form
        linear_part = self.W1 @ X + self.b
        result = V + self.W2 @ (linear_part * (self.W1 @ V))

        return result

    def jac_transpose_dx_mul_v(self, X, v_vector):
        V = v_vector  # reshape to wanted form
        linear_part = self.W1 @ X + self.b
        result = V + self.W1 @ (linear_part * (self.W2 @ V))

        return result

    def update_weights(self, learning_rate):
        self.W1 -= learning_rate * self.grad_W1
        self.W2 -= learning_rate * self.grad_W2
        self.b -= learning_rate * self.grad_b
