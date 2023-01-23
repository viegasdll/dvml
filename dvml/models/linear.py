"""
Module for linear models
"""
from dvml.models.model import SupervisedModel


class LinearRegression(SupervisedModel):
    """
    Placeholder for doc
    """

    def __init__(self):
        self.params = None

    def predict(self, x_pred):
        pass

    def train(self, x_train, y_train):
        pass

    def set_params(self, params):
        """Set model parameters to input `params`"""
        self.params = params
