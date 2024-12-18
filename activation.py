from enum import Enum
import numpy as np


class Activation(Enum):
    RELU = "relu",
    TANH = "tanh"

def get_activation(activation: Activation):
    if activation == Activation.RELU:
        return relu
    elif activation == Activation.TANH:
        return tanh
    else:
        raise ValueError(f"Unknown activation {activation}")

def get_activation_derivative(activation: Activation):
    if activation == Activation.RELU:
        return relu_derivative
    elif activation == Activation.TANH:
        return tanh_derivative
    else:
        raise ValueError(f"Unknown activation {activation}")

def relu(x):
    return np.maximum(x, 0)

def relu_derivative(x):
    return (x > 0).astype(float)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2