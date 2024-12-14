import numpy as np


def sample_minibatch(X, C, batch_size, is_samples_in_columns = False):
    if is_samples_in_columns:
        m = X.shape[1]
        indices = np.random.choice(m, batch_size, replace=False)
        X_batch, C_batch = X[:,indices], C[:,indices]
    else:
        m = X.shape[0]
        indices = np.random.choice(m, batch_size, replace=False)
        X_batch, C_batch = X[indices, ], C[indices,]

    return X_batch, C_batch


def sgd(X, C, initial_W, grad_f, batch_size, learning_rate = 0.1, max_iter = 10**6, tolerance = 1e-6, is_samples_in_columns = False):
    iter = 0
    theta = initial_W
    progression_list = [theta]

    while iter < max_iter:
        X_batch, C_batch = sample_minibatch(X, C, batch_size, is_samples_in_columns)
        g = grad_f(X_batch, C_batch, theta)
        if np.linalg.norm(g, ord=2) < tolerance:
            break

        theta = theta - learning_rate * g
        progression_list.append(theta)
        iter += 1

    return theta, progression_list

def test_sgd():
    pass

def sgd_with_momentum():
    pass

def rms_props():
    pass


