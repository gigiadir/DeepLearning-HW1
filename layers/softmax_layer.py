import numpy as np

from softmax_cross_entropy import softmax_cross_entropy_gradient_dx, softmax_cross_entropy_loss, \
    softmax_cross_entropy_gradient_dw, softmax
from utils.vector_utils import initialize_weights_vector

'''
X_raw : n x m
C : l x m
'''
class SoftmaxLayer:
    def __init__(self, input_dim, output_dim, w_vector = None):
        self.n = input_dim
        self.l = output_dim
        w_vector = w_vector if w_vector is not None else initialize_weights_vector(input_dim + 1, output_dim)

        self.X= None
        self.C = None
        self.output_grad = None
        self.loss = None
        self.result = None

        self.set_weights_and_bias_from_vector(w_vector)

    def set_weights_and_bias_from_vector(self, w_vector):
        self.w_vector = w_vector
        self.W = w_vector.reshape(self.n + 1, self.l, order='F')

    def get_weights_and_bias_vector(self):
        return self.w_vector

    def get_gradient_vector(self):
        return self.output_grad

    def forward(self, X, C):
        if X.shape[0] == self.n:
            m = X.shape[1]
            X = np.vstack((X, np.ones((1, m))))

        self.X = X
        self.C = C

        X_t_W = self.X.T @ self.W
        self.result = np.apply_along_axis(softmax, axis=1, arr=X_t_W)

        m = self.X.shape[1]
        loss = (-1 / m) * np.sum(self.C.T * np.log(self.result))

        return self.result, loss

    def backprop(self, _prev_gradient):
        gradient_dx = softmax_cross_entropy_gradient_dx(self.X, self.C, self.w_vector)
        self.output_grad = softmax_cross_entropy_gradient_dw(self.X, self.C, self.w_vector)

        return gradient_dx

    def update_weights(self, learning_rate):
        new_w_vector = self.w_vector - learning_rate * self.output_grad
        self.set_weights_and_bias_from_vector(new_w_vector)


    def get_loss_with_specific_weights(self, w_vector):
        old_w = self.w_vector
        self.set_weights_and_bias_from_vector(w_vector)

        _, loss = self.forward(self.X, self.C)
        self.set_weights_and_bias_from_vector(old_w)

        return loss

    def get_gradient_dw_at_weights(self, w_vector):
        old_w = self.w_vector
        old_output_grad = self.output_grad

        self.w_vector = w_vector
        self.backprop(None)
        output_grad = self.output_grad

        self.w_vector = old_w
        self.output_grad = old_output_grad

        return output_grad
