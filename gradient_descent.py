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

'''
    X : (n+1) x m
    C : (n+1) x m
    initial_W : (n+1) * l x 1
'''
def sgd(X, C, initial_W, grad_f, loss_func, batch_size, learning_rate = 0.1, max_epochs =1000, tolerance = 0.01, is_samples_in_columns = False):
    epoch = 0
    theta = initial_W
    theta_list = [theta]
    loss_list = []

    while epoch < max_epochs:
        X_batch, C_batch = sample_minibatch(X, C, batch_size, is_samples_in_columns)
        g = grad_f(X_batch, C_batch, theta)
        if np.linalg.norm(g, ord=2) < tolerance:
            break

        theta = theta - learning_rate * g
        loss = loss_func(X_batch, C_batch, theta)

        theta_list.append(theta)
        loss_list.append(loss)
        epoch += 1

    return theta, theta_list, loss_list

def test_sgd():
    pass

def sgd_with_momentum():
    pass

def rms_props():
    pass


