import unittest

import numpy as np

from dvml.models.linear import LinearRegression


class TestLinearRegression(unittest.TestCase):
    def test_set_params(self):
        model = LinearRegression()
        test_params = [0.1, 0.2, 0.3]
        formatted_params = np.array(test_params).reshape(3, 1)

        model.set_params(test_params)

        self.assertTrue(np.array_equal(model.params, formatted_params))

    def test_predict_error(self):
        model = LinearRegression()

        self.assertRaises(RuntimeError, model.predict, None)

    def _test_predict(self, y, x_pred, params):
        model = LinearRegression()
        model.set_params(params)

        y_form = np.array(y).reshape([len(y), 1])

        self.assertTrue(np.array_equal(model.predict(x_pred), y_form))

    def test_predict_01(self):
        self._test_predict([20], np.array([1, 2, 3]).reshape([1, 3]), [6, 1, 2, 3])

    def test_predict_02(self):
        self._test_predict([9, 3, 7], np.array([[0, 2], [-1, 1], [3, 0]]), [1, 2, 4])
