import numpy as np

from utils import reshape_weights_vector_to_matrix, flatten_weights_matrix_to_vector


def softmax(x: np.ndarray):
    z = x - max(x)
    result = np.exp(z) / np.sum(np.exp(z))

    return result

'''
    X : (n+1) x m
    C : l x m
    w_vector: (n+1) * l x 1   
'''
def softmax_cross_entropy(X, C, w_vector):
    W = reshape_weights_vector_to_matrix(X, C, w_vector)
    X_t_W = X.T @ W
    softmax_across_rows = np.apply_along_axis(softmax, axis =1, arr=X_t_W)
    log_softmax_X_t_W = np.log(softmax_across_rows)
    result = C.T * log_softmax_X_t_W

    m = X.shape[1]
    loss = (-1/m) * np.sum(result)

    return loss

'''
    X : (n+1) x m
    C : l x m
    w_vector: (n+1) * l x 1   
'''
def softmax_cross_entropy_gradient(X, C, w_vector):
    W = reshape_weights_vector_to_matrix(X, C, w_vector)
    X_t_W = X.T @ W
    softmax_across_rows = np.apply_along_axis(softmax, axis =1, arr=X_t_W)
    m = X.shape[1]

    gradient =  (1/m) * X @ (softmax_across_rows - C.T)
    gradient_vector = flatten_weights_matrix_to_vector(gradient)

    return gradient_vector
