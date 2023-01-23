"""
Module for linear models
"""
import numpy as np

from dvml.models.model import SupervisedGradientModel


class LinearRegression(SupervisedGradientModel):
    """
    Placeholder for doc
    """

    def __init__(self):
        self.params = None

    def predict(self, x_in):
        """
        Prediction function
        :param x_in:
        :return:
        """

        # First, check that the model is trained, or at least has parameters
        if self.params is None:
            raise RuntimeError("Model must be trained before predicting.")

        # Convert X to a NumPy array, and format for linear regression
        x_np = np.array(x_in)
        x_form = np.concatenate(
            (np.ones(len(x_np)).reshape([len(x_np), 1]), x_np), axis=1
        )

        return np.dot(x_form, self.params)

    def train(self, x_train, y_train):
        pass

    def set_params(self, params):
        """
        Sets the model parameters
        :param params:
        :return:
        """
        self.params = np.array(params)

    def loss(self, x_in, y_in, params_in=None):
        # Check if parameters were passed
        if params_in is None:
            params_loss = self.params
        # Otherwise, use internal parameters
        else:
            params_loss = np.array(params_in)

        # Convert X to a NumPy array, and format for linear regression
        x_np = np.array(x_in)
        x_form = np.concatenate(
            (np.ones(len(x_np)).reshape([len(x_np), 1]), x_np), axis=1
        )
        # Convert y to a NumPy vector
        y_form = np.array(y_in)

        # Compute prediction error
        diff = y_form - np.dot(x_form, params_loss)

        # Compute and return loss
        return 0.5 * np.inner(diff, diff)

    def gradient(self, x_in, y_in, params_in=None):
        # Check if parameters were passed
        if params_in is None:
            params_grad = self.params
        # Otherwise, use internal parameters
        else:
            params_grad = np.array(params_in)

        # Convert X to a NumPy array, and format for linear regression
        x_np = np.array(x_in)
        x_form = np.concatenate(
            (np.ones(len(x_np)).reshape([len(x_np), 1]), x_np), axis=1
        )
        # Convert y to a NumPy vector
        y_form = np.array(y_in)

        # Compute prediction error
        diff = y_form - np.dot(x_form, params_grad)

        # Compute and return gradient
        return np.dot(-x_form.transpose(), diff)
