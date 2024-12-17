import numpy as np

from utils.vector_utils import reshape_weights_vector_to_matrix


class Layer:

    def __init__(self, input_dim: int, output_dim:int, activation):
        self.W = np.random.randn(input_dim, output_dim)
        self.b = np.random.randn(output_dim, 1)
        self.activation = lambda x : np.tanh(x)
        self.activation_derivative = lambda x : 1 - np.tanh(x)**2
        self.input_dim = input_dim
        self.output_dim = output_dim


    def forward(self, x):
        self.input = x
        self.z = self.W * x + self.b #
        self.a = self.activation(self.z)

        return self.a

    '''
        #TODO: COMPLETE SIZE
        prev_gradient: 
    '''
    def backprop(self, prev_gradient):
        grad_theta = self.jac_transpose_v(self.input, prev_gradient)
        grad_b = grad_theta * self.activation_derivative(self.z)

        gradient = grad_theta, grad_b
        return gradient

    def jac_transpose_v(self, x, v):
        return np.dot(self.activation_derivative(self.z), v) @ x.T

    '''
        f : R*n -> R^m 
        v : (k*n) x 1 
        x : n x 1
        returns: 
    '''
    def jac_v(self, x, v):
        n = x.shape[0]
        k = v.shape[0] / n
        v_matrix = v.reshape(n, k, order='F')
        v_matrix_x = v_matrix @ x
        result = self.activation_derivative(self.z) * v_matrix_x

        return result



    def update_parameters(self, learning_rate):
        pass