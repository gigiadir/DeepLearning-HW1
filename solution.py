from functools import partial

from data_utils import create_dataset_training_data
from gradient_test import gradient_test
from softmax_cross_entropy import softmax_cross_entropy, softmax_cross_entropy_gradient


def section_1a():
    peaks_training_data = create_dataset_training_data("Peaks")
    X = peaks_training_data.X
    C = peaks_training_data.C.T
    n = peaks_training_data.n
    l = peaks_training_data.l

    softmax_cross_entropy_func = lambda w : softmax_cross_entropy(X, C, w)
    softmax_cross_entropy_gradient_func = lambda w : softmax_cross_entropy_gradient(X, C, w)

    gradient_test(softmax_cross_entropy_func, softmax_cross_entropy_gradient_func, (n+1) * l)

def section_1b():
    pass

def section_1c():
    pass

def main():
    section_1a()
    section_1b()
    section_1c()



if __name__ == '__main__':
    main()