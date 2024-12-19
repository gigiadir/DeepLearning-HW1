import numpy as np

from activation import Activation, get_activation, get_activation_derivative


class Layer:
    def __init__(self, input_dim: int, output_dim:int, activation: Activation, b = None, w_vector = None):
        self.b = b if b is not None else np.random.randn(output_dim, 1)
        self.activation = get_activation(activation)
        self.activation_derivative = get_activation_derivative(activation)
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.grad_b = None
        self.grad_w = None
        self.mb_size = None

        w_vector = w_vector if w_vector is not None else np.random.randn(output_dim * input_dim, 1)
        self.set_weights_vector(w_vector)

    def set_weights_vector(self, w_vector):
        self.w_vector = w_vector
        self.W = w_vector.reshape(self.output_dim, self.input_dim, order='F')

    '''
        x : input_dim x minibatch_size 
        b : output_dim x 1
        W : output_dim x input_dim
        output: output_dim x minibatch_size
    '''
    def forward(self, X, C):
        self.X = X
        self.mb_size = X.shape[1]
        linear_output = self.W @ X + self.b
        output = self.activation(linear_output)

        return output

    def backprop(self, next_grad_dx):
        grad_x = self.jac_transpose_dx_mul_v(self.X, next_grad_dx)

        self.grad_w = self.jac_transpose_dw_mul_v(self.W, next_grad_dx)
        self.grad_b = self.jac_transpose_db_mul_v(self.b, next_grad_dx)

        return grad_x

    def jac_dw_mul_v(self, w_vector, v_vector):
        W = w_vector.reshape(self.output_dim, self.input_dim, order='F')
        V = v_vector.reshape(self.output_dim, self.input_dim, order='F')

        linear_output = W @ self.X + self.b
        result = self.activation_derivative(linear_output) * (V @ self.X)

        return result

    '''
        w_vector : {output_dim * input_dim} x 1
        v :  output_dim x 1
        result : {output_dim * input_dim} x 1
    '''
    def jac_transpose_dw_mul_v(self, w_vector, v_vector):
        W = w_vector.reshape(self.output_dim, self.input_dim, order='F')
        V = v_vector.reshape(self.output_dim, self.mb_size, order='F')
        linear_output = W @ self.X + self.b
        result = (self.activation_derivative(linear_output) * V) @ self.X.T

        return result.flatten(order='F').reshape(-1, 1)

    '''
        b : output_dim x 1
        v_vector : output_dim x 1
        result : output_dim x 1
    '''
    def jac_db_mul_v(self, b, v_vector):
        V = np.tile(v_vector, self.mb_size)
        linear_output = self.W @ self.X + b
        result = self.activation_derivative(linear_output) * V

        return result

    '''
        b : output_dim x 1
        v_vector : output_dim x 1
        result : output_dim x 1
    '''
    def jac_transpose_db_mul_v(self, b, v_vector):
        V = v_vector.reshape(self.output_dim, self.mb_size, order='F')
        linear_output = self.W @ self.X + b
        result = self.activation_derivative(linear_output) * V
        result = result.sum(axis=1, keepdims = True)

        return result.flatten(order='F').reshape(-1, 1)

    '''
        x : input_dim x ms_size
        v : input_dim x ms_size
        result : output_dim x ms_size
    '''
    def jac_dx_mul_v(self, X, v_vector):
        X = X.reshape(self.input_dim, self.mb_size, order='F')
        V = v_vector.reshape(self.input_dim, self.mb_size, order='F')
        linear_output = self.W @ X + self.b
        result = self.activation_derivative(linear_output) * (self.W @ V)

        return result

    '''
        x : input_dim x ms_size
        v : output_dim x ms_size
        result : input_dim x ms_size
    '''
    def jac_transpose_dx_mul_v(self, X, v_vector):
        V = v_vector.reshape(self.output_dim, self.mb_size, order='F')
        linear_output = self.W @ X + self.b
        result = self.W.T @ (self.activation_derivative(linear_output) *  V)

        return result.flatten(order='F').reshape(-1, 1)


    def update_weights(self, learning_rate):
        self.b -= learning_rate * self.grad_b

        W_change = self.grad_w.reshape(self.output_dim, self.input_dim, order='F')
        self.W -= learning_rate * W_change

    def get_weights(self):
        result = np.hstack((self.W, self.b))
        return result.flatten(order='F').reshape(-1, 1)