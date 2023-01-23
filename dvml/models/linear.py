"""
Module for linear models
"""
import numpy as np

from dvml.models.model import SupervisedModel


class LinearRegression(SupervisedModel):
    """
    Placeholder for doc
    """

    def __init__(self):
        self.params = None

    def predict(self, x_pred):
        """
        Prediction function
        :param x_pred:
        :return:
        """

        # First, check that the model is trained, or at least has parameters
        if self.params is None:
            raise RuntimeError("Model must be trained before predicting.")

        # Convert the input to a NumPy array, and format for linear regression
        x_np = np.array(x_pred)
        x_in = np.concatenate(
            (np.ones(len(x_np)).reshape([len(x_np), 1]), x_np), axis=1
        )

        return np.dot(x_in, self.params)

    def train(self, x_train, y_train):
        pass

    def set_params(self, params):
        """
        Sets the model parameters
        :param params:
        :return:
        """
        self.params = np.array(params).reshape(len(params), 1)
