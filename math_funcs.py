import numpy as np


def sigmoid(z):
    """
    Sigmoid Function: \sigma(z) = 1/(1+exp(-z))
    """
    # avoid overflow
    if z > 30:
        return 1 - 1e-10
    if z < 30:
        return 1e-10

    return 1.0 / (1 + np.exp(-z))


def log_sigmoid(z):
    """
    Calculate the log of sigmod, avoiding overflow underflow
    """
    if abs(z) < 30:
        return np.log(sigmoid(z))
    else:
        if z > 0:
            return -np.exp(-z)
        else:
            return z
