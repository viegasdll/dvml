"""
Module for linear models
"""
import numpy as np

from dvml.models.model import SupervisedGradientModel
from dvml.optimization.gradient import GradientDescent
from dvml.utils.config_utils import parse_config


class LinearRegression(SupervisedGradientModel):
    """
    Linear regression model, trained with basic gradient descent
    """

    DEFAULT_TRAIN_CONF = {
        "gamma": 0.01,
        "n_iter": 1000,
        "verbose": False,
    }

    def __init__(self):
        self.params = None

    def predict(self, x_in):
        """
        Prediction function

        :param x_in: a pandas dataframe or numpy array of model features
        :return: numpy array of model predictions
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

    def train(self, x_train, y_train, conf: dict = None):
        """
        Trains the model using gradient descent

        :param x_train: a pandas dataframe or numpy array of model features
        :param y_train: target variable array
        :param conf: training configuration, dict-like object
            gamma (float): learning rate
            n_iter (int): number of iterations of gradient descent to run
            verbose (bool): whether to print intermediate results or not
        :return:
        """

        parsed_conf = parse_config(conf, self.DEFAULT_TRAIN_CONF)

        optimizer = GradientDescent(self)

        params_ini = np.zeros(len(x_train[0]) + 1)

        opt_params = optimizer.optimize(x_train, y_train, params_ini, parsed_conf)

        self.set_params(opt_params)

    def set_params(self, params):
        """
        Sets the model parameters

        :param params: array of desired model parameters
        :return:
        """
        self.params = np.array(params)

    def loss(self, x_in, y_in, params_in=None):
        """
        Loss function to be optimized when training the model.
        For linear regression, the loss is the residual sum of squares

        :param x_in: a pandas dataframe or numpy array of model features
        :param y_in: target variable array
        :param params_in: model parameters. If not passed, will use the trained model parameters
        :return: the residual sum of squares for the given input data
        """
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
