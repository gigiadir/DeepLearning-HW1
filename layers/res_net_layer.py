import numpy as np

from activation import Activation, get_activation, get_activation_derivative


class ResNetLayer:

    def __init__(self, dim, activation: Activation, w1_vector = None, w2_vector = None, b = None):
        self.activation = get_activation(activation)
        self.activation_derivative = get_activation_derivative(activation)
        self.n = dim

        self.grad_b = None
        self.grad_w1 = None
        self.grad_w2 = None
        self.mb_size = None

        w1_vector = w1_vector if w1_vector is not None else np.random.randn(dim * dim, 1)
        self.set_w1_vector(w1_vector)

        w2_vector = w2_vector if w2_vector is not None else np.random.randn(dim * dim, 1)
        self.set_w2_vector(w2_vector)

        b = b if b is not None else np.random.randn(dim, 1)
        self.set_bias_vector(b)


    def set_w1_vector(self, w1_vector):
        if w1_vector.size != self.n ** 2:
            raise ValueError("w1_vector size must be equal to n**2")

        self.w1_vector = w1_vector
        self.W1 = w1_vector.reshape(self.n, self.n, order='F')

    def set_w2_vector(self, w2_vector):
        if w2_vector.size != self.n ** 2:
            raise ValueError("w1_vector size must be equal to n**2")

        self.w2_vector = w2_vector
        self.W2 = w2_vector.reshape(self.n, self.n, order='F')

    def set_bias_vector(self, b):
        if b.shape != (self.n, 1):
            raise ValueError("b shape must be equal to n*1")

        self.b = b


    def forward(self, X, C):
        self.X = X
        self.mb_size = X.shape[1]

        linear_part = self.W1 @ X + self.b
        output = X + self.W2 @ self.activation(linear_part)

        return output


    def backward(self, next_grad_dx):
        grad_x = self.jac_transpose_dx_mul_v(self.X, next_grad_dx)

        self.grad_w1 = self.jac_transpose_dw1_mul_v(self.W1, next_grad_dx)
        self.grad_w2 = self.jac_transpose_dw2_mul_v(self.W2, next_grad_dx)
        self.grad_b = self.jac_transpose_db_mul_v(self.b, next_grad_dx)

        return grad_x

    def jac_db_mul_v(self, b, v_vector):
        V = np.tile(v_vector, self.mb_size)
        linear_part = self.W1 @ self.X + b
        result = self.W2 @ (self.activation_derivative(linear_part) * V)

        return result

    def jac_transpose_db_mul_v(self, b, v_vector):
        V = v_vector.reshape(self.n, self.mb_size, order='F')
        linear_part = self.W1 @ self.X + b
        result = self.activation_derivative(linear_part) * (self.W2.T @ V)
        result = result.sum(axis=1, keepdims=True)

        return result.flatten(order='F').reshape(-1, 1)

    def jac_dw1_mul_v(self, w1_vector, v_vector):
        W1 = w1_vector.reshape(self.n, self.n, order='F')
        V = v_vector.reshape(self.n, self.n, order='F')
        linear_part = W1 @ self.X + self.b
        result = self.W2 @ (self.activation_derivative(linear_part) * (V @ self.X))

        return result

    def jac_transpose_dw1_mul_v(self, w1_vector, v_vector):
        W1 = w1_vector.reshape(self.n, self.n, order='F')
        V = v_vector.reshape(self.n, self.mb_size, order='F')
        linear_part = W1 @ self.X + self.b
        result = (self.activation_derivative(linear_part) * (self.W2.T @ V)) @ self.X.T

        return result.flatten(order='F').reshape(-1, 1)

    def jac_dw2_mul_v(self, _W2, v_vector):
        linear_part = self.W1 @ self.X + self.b
        result = v_vector @ self.activation(linear_part)

        return result

    def jac_transpose_dw2_mul_v(self, _W2, v_vector):
        linear_part = self.W1 @ self.X + self.b
        result = v_vector @ self.activation(linear_part).T

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
