"""
Metrics for classification models
"""
import numpy as np


def accuracy(y_real, y_pred):
    """

    :param y_real:
    :param y_pred:
    :return:
    """
    yrf = np.array(y_real)
    ypf = np.array(y_pred)

    return np.sum((yrf - ypf) == 0) / len(y_real)
