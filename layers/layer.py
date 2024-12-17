import numpy as np

class Layer:

    def __init__(self, input_dim: int, output_dim:int):
        self.W = np.random.randn(output_dim, input_dim)
        self.b = np.random.randn(output_dim, 1)
        self.activation = lambda x : np.tanh(x)
        self.activation_derivative = lambda x : 1 - np.tanh(x)**2
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.x = None
        self.linear_output = None
        self.output = None


    '''
        x : input_dim x 1 
        b : output_dim x 1
        W : output_dim x input_dim
    '''
    def forward(self, x = None, W = None, b = None):
        x = x if x is not None else self.x
        W = W if W is not None else self.W
        b = b if b is not None else self.b

        self.x = x
        self.linear_output = W @ x + b
        self.output = self.activation(self.linear_output)

        return self.output

    def backward(self, prev_dW, prev_db):
        #calculate gradients
        # grad_W = None
        # grad_b = None
        #
        # self.b -= learning_rate * grad_b
        # self.W -= learning rate * grad_W
        #
        # return grad_W, grad_b
        pass

    '''
        w_vector : {output_dim * input_dim} x 1
        v_vector : {output_dim * input_dim} x 1
    '''
    def jac_dw_mul_v(self, w_vector, v_vector):
        W = w_vector.reshape(self.output_dim, self.input_dim, order='F')
        V = v_vector.reshape(self.output_dim, self.input_dim, order='F')

        linear_output = W @ self.x + self.b
        result = self.activation_derivative(linear_output) * (V @ self.x)

        return result

    '''
        W: k x n
        v : k x 1
        result : k x n
    '''
    def jac_transpose_dw_mul_v(self, w_vector, v):
        W = w_vector.reshape(self.output_dim, self.input_dim, order='F')
        linear_output = W @ self.x + self.b
        result = (self.activation_derivative(linear_output) * v) @ self.x.T

        return result.flatten(order='F')

    def jac_db_mul_v(self, b, v):
        linear_output = self.W @ self.x + b
        result = self.activation_derivative(linear_output) * v

        return result

    def jac_transpose_db_mul_v(self, b, v):
        linear_output = self.W @ self.x + b
        result = self.activation_derivative(linear_output) * v

        return result

    '''
        x : input_dim x 1 
        v : input_dim x 1
    '''
    def jac_dx_mul_v(self, x, v):
        linear_output = self.W @ x + self.b
        result = self.activation_derivative(linear_output) * (self.W @ v)

        return result

    def jac_transpose_dx_mul_v(self, x, v):
        linear_output = self.W @ x + self.b
        result = self.W.T @ (self.activation_derivative(linear_output) *  v)

        return result


    def update_parameters(self, learning_rate):
        pass