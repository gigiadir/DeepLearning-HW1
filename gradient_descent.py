import numpy as np

from data_utils import sample_minibatch

'''
    X : (n+1) x m
    C : l x m
    initial_W : (n+1) * l x 1
'''
def sgd(X, C, initial_W, batch_size,
        grad_f, loss_func, accuracy_func,
        learning_rate = 0.1, max_epochs = 1000, tolerance = 0.001, is_samples_in_columns = False):
    epoch = 0
    theta = initial_W
    loss_list, accuracy_list, theta_list = [], [], [theta]

    while epoch < max_epochs:
        X_batch, C_batch = sample_minibatch(X, C, batch_size, is_samples_in_columns)
        g = grad_f(X_batch, C_batch, theta)
        if np.linalg.norm(g, ord=2) < tolerance:
            break

        theta = theta - learning_rate * g
        loss = loss_func(X_batch, C_batch, theta)
        accuracy = accuracy_func(X, C, theta)

        loss_list.append(loss)
        accuracy_list.append(accuracy)
        theta_list.append(theta)

        if epoch % 100 == 0:
            print(f"epoch {epoch} - current loss: {loss}, current accuracy: {accuracy}")

        epoch += 1

    return theta, loss_list, accuracy_list, theta_list

def test_sgd():
    pass

def sgd_with_momentum():
    pass

def rms_props():
    pass


