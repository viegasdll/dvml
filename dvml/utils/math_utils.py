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


def gini_binary_avg(y_left, y_right):
    """
    Need to add the case when one is empty

    :param y_left:
    :param y_right:
    :return:
    """
    if y_left is None or len(y_left) == 0:
        return gini_binary(y_right)
    if y_right is None or len(y_right) == 0:
        return gini_binary(y_left)

    gini_left = gini_binary(y_left)
    gini_right = gini_binary(y_right)

    p_left = len(y_left) / (len(y_left) + len(y_right))
    p_right = len(y_right) / (len(y_left) + len(y_right))

    return p_left * gini_left + p_right * gini_right


def gini_binary_split(x_in, y_in, boundary):
    """
    Computes the gini impurity of a split along a given boundary

    :param x_in:
    :param y_in:
    :param boundary:
    :return:
    """
    y_left = [y_val for (x_val, y_val) in zip(x_in, y_in) if x_val < boundary]
    y_right = [y_val for (x_val, y_val) in zip(x_in, y_in) if x_val >= boundary]

    return gini_binary_avg(y_left, y_right)


def gini_opt_split(x_in, y_in):
    """
    Tries to find a good split of the data that minimizes gini impurity
    Started with a very naive approach, can improve over time

    :param x_in:
    :param y_in:
    :return: the best boundary found, and the resulting gini impurity
    """
    n_boundaries = 10

    x_min = min(x_in)
    x_max = max(x_in)

    # If constant, return the value itself
    if x_min == x_max:
        return x_min, gini_binary_split(x_in, y_in, x_min)

    # If not, define a set of candidate boundaries
    boundaries = np.arange(
        x_min + 1 / n_boundaries, x_max, (x_max - x_min) / n_boundaries
    )

    # For each boundary, find the resulting gini impurity from the split
    split_ginis = [gini_binary_split(x_in, y_in, bdr) for bdr in boundaries]

    # Return the best split, that minimizes the gini impurity
    best_split_ind = split_ginis.index(min(split_ginis))

    return boundaries[best_split_ind], split_ginis[best_split_ind]
