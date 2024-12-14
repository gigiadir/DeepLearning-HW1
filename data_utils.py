import scipy.io
import os
import numpy as np

from models.dataset_training_data import DatasetTrainingData


def load_dataset(dataset_name):
    data_path = os.path.join("./Data", f"{dataset_name}Data.mat")
    mat_data = scipy.io.loadmat(data_path)

    datasets = {
        'X_train': mat_data["Yt"],
        'C_train': mat_data["Ct"],
        'X_validation': mat_data["Yv"],
        'C_validation': mat_data["Cv"]
    }

    return datasets

def create_dataset_training_data(dataset_name):
    dataset = load_dataset(dataset_name)
    X_raw, C = dataset['X_train'], dataset['C_train']
    training_data = DatasetTrainingData(X_raw, C)

    return training_data

