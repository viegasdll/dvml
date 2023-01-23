import unittest

import numpy as np

from dvml.optimization.gradient import GradientDescent
from dvml.models.linear import LinearRegression


class TestGradientDescent(unittest.TestCase):
    def test_gradient_descent(self):
        x_in = [[1, -1], [-2, 4]]
        y_in = [1, 14]
        params_ini = [2, -1, 3]

        model = LinearRegression()
        grad_desc = GradientDescent(model)

        params = grad_desc.optimize(x_in, y_in, params_ini, gamma=.05, n_iter=120, verbose=False)

        loss = model.loss(x_in, y_in, params)

        self.assertAlmostEqual(loss, 0)

    def test_gradient_descent_v(self):
        x_in = [[1, -1], [-2, 4]]
        y_in = [1, 14]
        params_ini = [2, -1, 3]

        model = LinearRegression()
        grad_desc = GradientDescent(model)

        params = grad_desc.optimize(x_in, y_in, params_ini, gamma=.05, n_iter=120)

        loss = model.loss(x_in, y_in, params)

        self.assertAlmostEqual(loss, 0)
