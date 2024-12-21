import scipy.io
import os
import numpy as np

from models.dataset_training_data import DatasetTrainingData

'''
    X : (n+1) x m or m x (n+1)
    C : l x m or m x l   
'''
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


def load_dataset(dataset_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "../data", f"{dataset_name}Data.mat")
    mat_data = scipy.io.loadmat(data_path)

    datasets = {
        'X_train': mat_data["Yt"],
        'C_train': mat_data["Ct"],
        'X_validation': mat_data["Yv"],
        'C_validation': mat_data["Cv"]
    }

    return datasets

def create_dataset_training_data(X_raw, C, percentage = 1):
    if percentage < 1:
        m = X_raw.shape[1]
        X_raw, C = sample_minibatch(X_raw, C, int(percentage * m), is_samples_in_columns=True)

    training_data = DatasetTrainingData(X_raw, C)

    return training_data

def get_dataset(dataset_name, percentage = 1):
    dataset = load_dataset(dataset_name)
    training_data = create_dataset_training_data(dataset["X_train"], dataset["C_train"], percentage)
    validation_data = create_dataset_training_data(dataset["X_validation"], dataset["C_validation"], percentage)

    return training_data, validation_data

