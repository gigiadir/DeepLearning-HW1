import numpy as np


''' 
    m: number of examples
    n: number of features
    l: number of labels
    X_raw: (n x m) - the original data
    C: (l x m)
    W_raw:  (n x l) - the original weights
    X: ((n+1), m) - the original data + row of ones
    W: ((n+1), l) - the original weights + biases for each label
'''

class DatasetTrainingData:
    X_raw: np.ndarray
    C: np.ndarray
    X: np.ndarray
    m: int
    n: int
    l: int
    W_raw: np.ndarray
    W: np.ndarray

    def __init__(self, X_raw: np.ndarray, C: np.ndarray):
        self.X_raw = X_raw
        self.C = C
        self.init_dimensions()
        self.init_weights_and_biases()

    def init_dimensions(self):
        self.n = self.X_raw.shape[0]
        self.m = self.X_raw.shape[1]
        self.l = self.C.shape[1]

    def init_weights_and_biases(self):
        X_raw = self.X_raw
        row_of_ones = np.ones((1, self.m))
        X = np.vstack((X_raw, row_of_ones))

        W_raw = np.random.randn(self.n, self.l)
        random_row = np.random.rand(1, self.l)
        W = np.vstack((W_raw, random_row))

        self.W_raw = W_raw
        self.X = X
        self.W = W


