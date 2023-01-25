"""
Module for gradient-descent optimizers
"""
from dvml.models.model import SupervisedGradientModel


class GradientDescent:  # pylint: disable=too-few-public-methods
    """
    Basic gradient descent optimizer
    """

    DEFAULT_CONF = {
        "gamma": 0.01,
        "n_iter": 1000,
        "verbose": True,
    }

    def __init__(self, model: SupervisedGradientModel):
        self.model = model

    def optimize(self, x_in, y_in, params_ini, conf: dict = None):
        """

        :param x_in:
        :param y_in:
        :param params_ini:
        :param conf:
        :return:
        """
        if conf is None:
            conf_def = self.DEFAULT_CONF
        else:
            conf_def = {}
            # Validate options, or set to default
            for opt, default_val in self.DEFAULT_CONF.items():
                conf_def[opt] = conf.get(opt, default_val)

        print("Starting gradient descent optimization...")
        loss_ini = self.model.loss(x_in, y_in, params_ini)
        print(f"Initial loss: {loss_ini}")

        params = params_ini

        for ind in range(conf_def["n_iter"]):
            grad = self.model.gradient(x_in, y_in, params)
            params = params - (conf_def["gamma"] * grad)
            loss = self.model.loss(x_in, y_in, params)
            if conf_def["verbose"]:
                print(f"[{ind+1}/{conf_def['n_iter']}] Loss: {loss}")

        print(f"Final loss: {loss}")

        return params
