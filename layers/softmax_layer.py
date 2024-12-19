import numpy as np

from softmax_cross_entropy import softmax_cross_entropy_gradient_dx, softmax_cross_entropy_loss, \
    softmax_cross_entropy_gradient_dw

'''
X_raw : n x m
C : l x m
'''
class SoftmaxLayer:
    def __init__(self, input_dim, output_dim):
        self.n = input_dim
        self.l = output_dim
        self.w_vector = np.random.randn((input_dim + 1) * output_dim, 1)

        self.X= None
        self.C = None
        self.output_grad = None

    def forward(self, X, C, w_vector = None):
        old_w = self.w_vector

        if w_vector is not None:
            self.w_vector = w_vector

        m = X.shape[1]
        X = np.vstack((X, np.ones((1, m))))

        loss = softmax_cross_entropy_loss(X, C, self.w_vector)

        self.X = X
        self.C = C
        self.w_vector = old_w

        return loss

    def backprop(self, _prev_gradient):
        gradient_dx = softmax_cross_entropy_gradient_dx(self.X, self.C, self.w_vector)
        self.output_grad = softmax_cross_entropy_gradient_dw(self.X, self.C, self.w_vector)

        return gradient_dx

    def update_weights(self, learning_rate):
        self.w_vector -= learning_rate * self.output_grad

    def get_weights(self):
        return self.output_grad
