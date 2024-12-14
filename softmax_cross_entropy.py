import numpy as np

from utils import reshape_weights_vector_to_matrix


def softmax(x: np.ndarray):
    z = x - max(x)
    result = np.exp(z) / np.sum(np.exp(z))

    return result

'''
    X : (n+1) x m
    C : m x l
    w_vector: (n+1) * l x 1   
'''
def softmax_cross_entropy(X, C, w_vector):
    W = reshape_weights_vector_to_matrix(X, C, w_vector)
    X_t_W = X.T @ W
    softmax_across_rows = np.apply_along_axis(softmax, axis =1, arr=X_t_W)
    log_softmax_X_t_W = np.log(softmax_across_rows)
    result = C * log_softmax_X_t_W

    m = X.shape[1]
    loss = (-1/m) * np.sum(result)

    return loss

'''
    X : n x m
    C : m x l
    w_vector: (n+1) * l x 1   
'''
def softmax_cross_entropy_gradient(X, C, w_vector):
    W = reshape_weights_vector_to_matrix(X, C, w_vector)
    X_t_W = X.T @ W
    softmax_across_rows = np.apply_along_axis(softmax, axis =1, arr=X_t_W)
    m = X.shape[1]

    gradient =  (1/m) * X @ (softmax_across_rows - C)
    gradient_vector = gradient.flatten(order = 'F')

    return gradient_vector
