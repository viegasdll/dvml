"""
Utility functions for math operations
"""
import numpy as np


def sigmoid(x_in):
    """
    Computes the sigmoid function for each value of an array

    :param x_in: numpy array (or equivalent)
    :return: numpy array with value of sigmoid function for each element of x_in
    """
    sig = 1 / (1 + np.exp(-np.array(x_in)))

    return sig


def gini_binary(y_in):
    """
    Gini impurity for binary classification. It assumes the label is either 0 or 1

    :param y_in: 1-D numpy array (or equivalent)
    :return: gini impurity of the input vector
    """
    pr1 = np.sum(y_in) / len(y_in)
    pr0 = 1 - pr1

    return 1 - (pr0**2) - (pr1**2)
