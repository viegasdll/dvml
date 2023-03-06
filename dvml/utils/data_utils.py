"""
Utility functions to handle data
"""
import random

import numpy as np


def parse_x_lr(x_in):
    """
    Parse features data for linear regression, by adding a column of 1s to the left (for intercept)

    :param x_in: numpy array (or equivalent) with the input features
    :return:
    """
    try:
        x_np = np.array(x_in)
        ones = np.ones(len(x_np)).reshape([len(x_np), 1])
    except Exception as exc:
        raise TypeError("The data must be numpy array or equivalent") from exc

    return np.concatenate((ones, x_np), axis=1)


def parse_x_ct(x_in):
    """
    Parse features data for classification tree

    :param x_in: numpy array (or equivalent) with the input features
    :return:
    """
    # Convert x_in to a numpy array
    x_form = np.array(x_in)
    # Make it into a matrix if it was just a vector
    if x_form.ndim == 1:
        x_form = x_form.reshape([1, len(x_form)])

    return x_form


def bootstrap(x_in, y_in, n_samples=None):
    """
    Creates a bootstrap sample for model training

    :param n_samples: desired output size, can be smaller or larger than input
    :param x_in: numpy array (or equivalent) with the input features
    :param y_in: 1-D numpy array with the input target variable
    :return: x_out, y_out - bootstrap samples with desired size
    """
    if n_samples is None:
        n_samples = len(y_in)

    x_out = np.zeros([n_samples, x_in.shape[1]])
    y_out = np.zeros(n_samples)

    for int_out in range(n_samples):
        ind_in = random.randint(0, n_samples - 1)

        x_out[int_out] = x_in[ind_in]
        y_out[int_out] = y_in[ind_in]

    return x_out, y_out
