"""
Module for gradient-descent optimizers
"""
from dvml.models.model import SupervisedGradientModel


class GradientDescent:  # pylint: disable=too-few-public-methods
    """
    Basic gradient descent optimizer
    """

    def __init__(self, model: SupervisedGradientModel):
        self.model = model

    def optimize(
        self, x_in, y_in, params_ini, gamma=0.01, n_iter=1000, verbose=True
    ):  # pylint: disable=too-many-arguments
        """

        :param x_in:
        :param y_in:
        :param params_ini:
        :param gamma:
        :param n_iter:
        :param verbose:
        :return:
        """
        print("Starting gradient descent optimization...")
        loss_ini = self.model.loss(x_in, y_in, params_ini)
        print(f"Initial loss: {loss_ini}")

        params = params_ini

        for ind in range(n_iter):
            grad = self.model.gradient(x_in, y_in, params)
            params = params - (gamma * grad)
            loss = self.model.loss(x_in, y_in, params)
            if verbose:
                print(f"[{ind+1}/{n_iter}] Loss: {loss}")

        print(f"Final loss: {loss}")

        return params
