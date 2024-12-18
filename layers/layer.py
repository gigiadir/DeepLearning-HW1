import numpy as np

from activation import Activation, get_activation, get_activation_derivative


class Layer:

    def __init__(self, input_dim: int, output_dim:int, activation: Activation):
        self.W = np.random.randn(output_dim, input_dim)
        self.b = np.random.randn(output_dim, 1)
        self.activation = get_activation(activation)
        self.activation_derivative = get_activation_derivative(activation)
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.x = None
        self.linear_output = None
        self.output = None
        self.grad_b = None
        self.grad_w = None


    '''
        x : input_dim x 1 
        b : output_dim x 1
        W : output_dim x input_dim
    '''
    def forward(self, x = None, W = None, b = None, C = None):
        x = x if x is not None else self.x
        W = W if W is not None else self.W
        b = b if b is not None else self.b

        self.x = x
        self.linear_output = W @ x + b
        self.output = self.activation(self.linear_output)

        return self.output, C

    def backward(self, next_grad_dx):
        grad_x = self.jac_transpose_dx_mul_v(self.x, next_grad_dx)

        self.grad_w = self.jac_transpose_dw_mul_v(self.W, next_grad_dx)
        self.grad_b = self.jac_transpose_db_mul_v(self.b, next_grad_dx)

        return grad_x

    '''
        w_vector : {output_dim * input_dim} x 1
        v_vector : {output_dim * input_dim} x 1
        result : output_dim * 1
    '''
    def jac_dw_mul_v(self, w_vector, v_vector):
        W = w_vector.reshape(self.output_dim, self.input_dim, order='F')
        V = v_vector.reshape(self.output_dim, self.input_dim, order='F')

        linear_output = W @ self.x + self.b
        result = self.activation_derivative(linear_output) * (V @ self.x)

        return result

    '''
        w_vector : {output_dim * input_dim} x 1
        v :  output_dim x 1
        result : {output_dim * input_dim} x 1
    '''
    def jac_transpose_dw_mul_v(self, w_vector, v):
        W = w_vector.reshape(self.output_dim, self.input_dim, order='F')
        linear_output = W @ self.x + self.b
        result = (self.activation_derivative(linear_output) * v) @ self.x.T

        return result.flatten(order='F')

    '''
        b : output_dim x 1
        v : output_dim x 1
        result : output_dim x 1
    '''
    def jac_db_mul_v(self, b, v):
        linear_output = self.W @ self.x + b
        result = self.activation_derivative(linear_output) * v

        return result

    '''
        b : output_dim x 1
        v : output_dim x 1
        result : output_dim x 1
    '''
    def jac_transpose_db_mul_v(self, b, v):
        linear_output = self.W @ self.x + b
        result = self.activation_derivative(linear_output) * v

        return result

    '''
        x : input_dim x 1 
        v : input_dim x 1
        result : output_dim x 1
    '''
    def jac_dx_mul_v(self, x, v):
        linear_output = self.W @ x + self.b
        result = self.activation_derivative(linear_output) * (self.W @ v)

        return result

    '''
        x : input_dim x 1
        v : output_dim x 1
        result : input_dim x 1 
    '''
    def jac_transpose_dx_mul_v(self, x, v):
        linear_output = self.W @ x + self.b
        result = self.W.T @ (self.activation_derivative(linear_output) *  v)

        return result


    def update_weights(self, learning_rate):
        self.b -= learning_rate * self.grad_b
        self.W -= learning_rate * self.grad_w