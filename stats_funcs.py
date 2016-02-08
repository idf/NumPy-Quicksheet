import numpy as np
from scipy.stats import logistic, multivariate_normal


def sigmoid(X):
    return logistic.cdf(X)


def sigmoid_deprecated(z):
    """
    Sigmoid Function: \sigma(z) = 1/(1+exp(-z))
    """
    # avoid overflow
    if z > 30:
        return 1 - 1e-10
    if z < 30:
        return 1e-10

    return 1.0 / (1 + np.exp(-z))


def log_sigmoid_deprecated(z):
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


def multivariate_gaussian(X, mu, Sigma):
    return multivariate_normal.pdf(X, mean=mu, cov=Sigma)
    
