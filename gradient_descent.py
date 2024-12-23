import numpy as np

from utils.data_utils import sample_minibatch

'''
    X : (n+1) x m
    C : l x m
    initial_W : (n+1) * l x 1
'''
def sgd(X, C, initial_W, batch_size,
        grad_f, loss_func, accuracy_func = None,
        learning_rate = 0.01, max_iterations =1000, tolerance = 1e-4, is_samples_in_columns = False):
    iteration = 0
    theta = initial_W
    loss_list, accuracy_list, theta_list = [], [], [theta]

    while iteration < max_iterations:
        X_batch, C_batch = sample_minibatch(X, C, batch_size, is_samples_in_columns)
        g = grad_f(X_batch, C_batch, theta)
        if np.linalg.norm(g, ord=2) < tolerance:
            break

        theta = theta - learning_rate * g
        loss = loss_func(X_batch, C_batch, theta)
        loss_list.append(loss)

        if accuracy_func:
            accuracy = accuracy_func(X, C, theta)
            accuracy_list.append(accuracy)

        theta_list.append(theta)

        if iteration % 100 == 0:
            print(f"iteration {iteration} - current loss: {np.squeeze(loss)}, current accuracy: {accuracy if accuracy_func is not None else 'Not Tested'}")

        iteration += 1

    return theta, loss_list, accuracy_list, theta_list


def sgd_with_momentum(X, C, initial_W, batch_size,
                      grad_f, loss_func, accuracy_func = None, learning_rate=0.01,
                      momentum = 0.9, max_epochs=1000, tolerance=1e-4, is_samples_in_columns=False):
        epoch = 1
        theta = initial_W
        loss_list, accuracy_list, theta_list = [], [], []
        v = np.zeros_like(theta)
        num_samples = X.shape[1]
        num_minibatches = int(num_samples / batch_size)
        while epoch < max_epochs:
            shuffled_indices = np.random.permutation(num_samples)
            for i in range(0, num_samples, batch_size):
                loss = 0
                indices = shuffled_indices[i:i + batch_size]
                X_batch, C_batch = X[:, indices], C[:, indices]
                g = grad_f(X_batch, C_batch, theta)
                if np.linalg.norm(g, ord=2) < tolerance:
                    break

                v = momentum * v - learning_rate * g
                theta = theta + v

                loss += loss_func(X_batch, C_batch, theta)

            loss_list.append(loss/num_minibatches)

            if accuracy_func:
                accuracy = accuracy_func(X, C, theta)
                accuracy_list.append(accuracy)

            theta_list.append(theta)

            if epoch % 100 == 0:
                print(f"epoch {epoch} - current loss: {np.squeeze(loss)}, current accuracy: {accuracy if accuracy_func is not None else 'Not Tested'}")

            epoch += 1

        return theta, loss_list, accuracy_list, theta_list

def rms_props():
    pass


