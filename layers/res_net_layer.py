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

    def backprop(self, next_grad_dx):
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

    def jac_dw2_mul_v(self, _w2_vector, v_vector):
        V = v_vector.reshape(self.n, self.n, order='F')
        linear_part = self.W1 @ self.X + self.b
        result = V @ self.activation(linear_part)

        return result

    def jac_transpose_dw2_mul_v(self, _w2_vector, v_vector):
        V = v_vector.reshape(self.n, self.mb_size, order='F')
        linear_part = self.W1 @ self.X + self.b
        result = V @ (self.activation(linear_part)).T

        return result.flatten(order='F').reshape(-1, 1)

    def jac_dx_mul_v(self, x_vector, v_vector):
        X = x_vector.reshape(self.n, self.mb_size, order='F')
        V = v_vector.reshape(self.n, self.mb_size, order='F')
        linear_part = self.W1 @ X + self.b
        result = V + self.W2 @ (self.activation_derivative(linear_part) * (self.W1 @ V))

        return result

    def jac_transpose_dx_mul_v(self, X, v_vector):
        V = v_vector.reshape(self.n, self.mb_size, order='F')
        linear_part = self.W1 @ X + self.b
        result = V + self.W1.T @ (self.activation_derivative(linear_part) * (self.W2.T @ V))

        return result.flatten(order='F').reshape(-1, 1)

    def update_weights(self, learning_rate):
        W1_change = self.grad_w1.reshape(self.n, self.n, order='F')
        self.W1 -= learning_rate * W1_change

        W2_change = self.grad_w2.reshape(self.n, self.n, order='F')
        self.W2 -= learning_rate * W2_change

        self.b -= learning_rate * self.grad_b

    def get_gradient_vector(self):
        gradient_vector = np.concatenate((self.grad_w1, self.grad_w2, self.grad_b), axis=0)

        return gradient_vector

    def set_weights_and_bias_from_vector(self, vector):
        if vector.shape != (2*self.n**2 + self.n, 1):
            raise ValueError("The weights and bias vector must be of shape 2*n^2 + 1.")

        w1_vector_split_index = self.n ** 2
        w1_vector = vector[:w1_vector_split_index]
        w2_vector_split_index = 2 * self.n ** 2
        w2_vector = vector[w1_vector_split_index:w2_vector_split_index]
        b_vector = vector[w2_vector_split_index:]

        self.set_w1_vector(w1_vector)
        self.set_w2_vector(w2_vector)
        self.set_bias_vector(b_vector)

    def get_weights_and_bias_vector(self):
        weights_and_bias_vector = np.concatenate((self.w1_vector, self.w2_vector, self.b), axis=0)

        return weights_and_bias_vector
