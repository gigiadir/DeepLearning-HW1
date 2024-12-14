'''
reshapes vector from size (n+1) * l to (n+1) x l matrix by columns
'''
def reshape_weights_vector_to_matrix(X, C, w_vector):
    w_rows, w_cols = X.shape[0], C.shape[1]
    W = w_vector.reshape(w_rows, w_cols, order='F')

    return W