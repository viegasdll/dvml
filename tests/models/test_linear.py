import unittest

import numpy as np
from sklearn.datasets import load_diabetes

from dvml.models.linear import LinearRegression


class TestLinearRegression(unittest.TestCase):
    def test_set_params(self):
        model = LinearRegression()
        test_params = [0.1, 0.2, 0.3]
        formatted_params = np.array(test_params)

        model.set_params(test_params)

        self.assertTrue(np.array_equal(model.params, formatted_params))

    def test_predict_error(self):
        model = LinearRegression()

        self.assertRaises(RuntimeError, model.predict, None)

    def _test_predict(self, y, x_in, params):
        model = LinearRegression()
        model.set_params(params)

        y_form = np.array(y)

        self.assertTrue(np.array_equal(model.predict(x_in), y_form))

    def test_predict_01(self):
        self._test_predict([20], np.array([1, 2, 3]).reshape([1, 3]), [6, 1, 2, 3])

    def test_predict_02(self):
        self._test_predict([9, 3, 7], np.array([[0, 2], [-1, 1], [3, 0]]), [1, 2, 4])

    def _test_loss(self, y, x_in, params_mod, params_in, expected_loss):
        model = LinearRegression()
        model.set_params(params_mod)

        loss = model.loss(x_in, y, params_in)

        self.assertEquals(loss, expected_loss)

    def test_loss_01(self):
        self._test_loss([1, 14], [[1, -1], [-2, 4]], None, [2, -1, 3], 6.5)

    def test_loss_02(self):
        self._test_loss([1, 14], [[1, -1], [-2, 4]], [2, -1, 3], None, 6.5)

    def _test_gradient(self, y, x_in, params_mod, params_in, expected_grad):
        model = LinearRegression()
        model.set_params(params_mod)

        grad = model.gradient(x_in, y, params_in)

        self.assertTrue(np.array_equal(grad, expected_grad))

    def test_gradient_01(self):
        self._test_gradient([1, 14], [[1, -1], [-2, 4]], None, [2, -1, 3], [-1, -7, 11])

    def test_gradient_02(self):
        self._test_gradient([1, 14], [[1, -1], [-2, 4]], [2, -1, 3], None, [-1, -7, 11])

    def test_train(self):
        model = LinearRegression()

        x_train = [[1, -1], [-2, 4]]
        y_train = [1, 14]

        model.train(x_train, y_train)

        loss = model.loss(x_train, y_train)

        self.assertAlmostEqual(loss, 0)

    def test_train_conf(self):
        model = LinearRegression()

        x_train = [[1, -1], [-2, 4]]
        y_train = [1, 14]

        conf = {
            "gamma": 0.05,
            "n_iter": 150,
        }

        model.train(x_train, y_train, conf)

        loss = model.loss(x_train, y_train)

        self.assertAlmostEqual(loss, 0)

    def test_e2e(self):
        model = LinearRegression()

        diabetes_dataset = load_diabetes()
        x_train = diabetes_dataset.data
        y_train = diabetes_dataset.target

        conf = {
            "gamma": 0.002,
            "verbose": False,
            "n_iter": 2000,
        }

        loss_start = model.loss(x_train, y_train, np.zeros(x_train.shape[1] + 1))

        model.train(x_train, y_train, conf)
        loss_end = model.loss(x_train, y_train)

        loss_ratio = loss_end / loss_start

        self.assertLess(loss_ratio, 0.1)
