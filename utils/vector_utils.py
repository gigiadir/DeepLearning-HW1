'''
reshapes vector from size (n+1) * l to (n+1) x l matrix by columns
'''
import numpy as np


def reshape_weights_vector_to_matrix(X, C, w_vector):
    w_rows, w_cols = X.shape[0], C.shape[0]
    W = w_vector.reshape(w_rows, w_cols, order='F')

    return W


def flatten_weights_matrix_to_vector(w_matrix):
    w_vector = w_matrix.flatten(order='F').reshape(-1, 1)

    return w_vector


def initialize_weights_vector(input_dim, output_dim):
    std = np.sqrt(2 / (input_dim + output_dim))

    W = np.random.normal(loc=0.0, scale=std, size=(input_dim, output_dim))
    w_vector = W.flatten(order='F').reshape(-1, 1)

    return w_vector