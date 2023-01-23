"""
Module with the template classes for models
"""
from abc import ABC, abstractmethod

import numpy as np


class Model(ABC):  # pylint: disable=too-few-public-methods
    """
    Generic model class
    """

    @abstractmethod
    def predict(self, x_in):
        """
        Prediction function
        :param x_in: the input dataset, a pandas dataframe or numpy array
        :return:
        """


class SupervisedModel(Model):
    """
    Generic supervised model class
    """

    @abstractmethod
    def train(self, x_train, y_train, conf: dict = None):
        """

        :param x_train:
        :param y_train:
        :param conf:
        :return:
        """


class SupervisedGradientModel(SupervisedModel):
    """
    Generic supervised model class,
    With a differentiable loss function
    """

    @abstractmethod
    def loss(self, x_in, y_in, params_in=None):
        """

        :param x_in:
        :param y_in:
        :param params_in:
        :return:
        """

    @abstractmethod
    def gradient(self, x_in, y_in, params_in=None):
        """

        :param x_in:
        :param y_in:
        :param params_in:
        :return:
        """


class UnsupervisedModel(Model):
    """
    Generic supervised model class
    """

    @abstractmethod
    def train(self, x_train):
        """

        :param x_train:
        :return:
        """


class ConstantModel(Model):  # pylint: disable=too-few-public-methods
    """
    Constant model: dummy model for basic tests.
    """

    def __init__(self, y_val):
        self.y_val = y_val

    def predict(self, x_in):
        return np.ones(len(x_in)) * self.y_val
