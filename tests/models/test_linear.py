import unittest

import numpy as np
from sklearn.datasets import load_diabetes, load_iris
from numpy.testing import assert_array_almost_equal

from dvml.models.linear import LinearRegression, LogisticRegression


class TestLinearModel(unittest.TestCase):
    def _test_gradient(self, y, x_in, params_mod, params_in, expected_grad):
        model = LinearRegression()
        model.set_params(params_mod)

        grad = model.gradient(x_in, y, params_in)

        self.assertTrue(np.array_equal(grad, expected_grad))

    def test_gradient_01(self):
        self._test_gradient([1, 14], [[1, -1], [-2, 4]], None, [2, -1, 3], [-1, -7, 11])

    def test_gradient_02(self):
        self._test_gradient([1, 14], [[1, -1], [-2, 4]], [2, -1, 3], None, [-1, -7, 11])

    def test_set_params(self):
        model = LinearRegression()
        test_params = [0.1, 0.2, 0.3]
        formatted_params = np.array(test_params)

        model.set_params(test_params)

        self.assertTrue(np.array_equal(model.params, formatted_params))


class TestLinearRegression(unittest.TestCase):
    def _test_loss(self, y, x_in, params_mod, params_in, expected_loss):
        model = LinearRegression()
        model.set_params(params_mod)

        loss = model.loss(x_in, y, params_in)

        self.assertEquals(loss, expected_loss)

    def test_loss_01(self):
        self._test_loss([1, 14], [[1, -1], [-2, 4]], None, [2, -1, 3], 6.5)

    def test_loss_02(self):
        self._test_loss([1, 14], [[1, -1], [-2, 4]], [2, -1, 3], None, 6.5)

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


class TestLogisticRegression(unittest.TestCase):
    def test_predict_error(self):
        model = LogisticRegression()

        self.assertRaises(RuntimeError, model.predict, None)

    def _test_predict(self, y, x_in, params):
        model = LogisticRegression()
        model.set_params(params)

        y_form = np.array(y)

        assert_array_almost_equal(model.predict(x_in), y_form)

    def test_predict_01(self):
        x_in = [[0, 0, 0], [-1000, -1000, -1000], [1000, 1000, 1000]]

        expected = [
            0.5,
            0,
            1,
        ]

        params = [
            0,
            1,
            1,
            1,
        ]

        self._test_predict(expected, x_in, params)

    def _test_loss(self, y, x_in, params_mod, params_in, expected_loss):
        model = LogisticRegression()
        model.set_params(params_mod)

        loss = model.loss(x_in, y, params_in)

        assert_array_almost_equal(loss, expected_loss)

    def test_loss_01(self):
        y = [
            0,
            1,
        ]

        params_mod = [
            0,
            1,
            1,
            1,
        ]

        params_in = None

        x_in = [[-1000, -1000, -1000], [1000, 1000, 1000]]

        expected_loss = 0

        self._test_loss(y, x_in, params_mod, params_in, expected_loss)

    def test_loss_02(self):
        y = [
            0,
            1,
        ]

        params_in = [
            0,
            1,
            1,
            1,
        ]

        params_mod = None

        x_in = [[-1000, -1000, -1000], [1000, 1000, 1000]]

        expected_loss = 0

        self._test_loss(y, x_in, params_mod, params_in, expected_loss)

    def test_loss_03(self):
        y = [
            0,
            1,
        ]

        params_in = [
            0,
            1,
            1,
            1,
        ]

        params_mod = None

        x_in = [[-1000, -1000, -1000], [-1000, -1000, -1000]]

        expected_loss = 15.249237733900

        self._test_loss(y, x_in, params_mod, params_in, expected_loss)

    def test_loss_04(self):
        y = [
            0,
            1,
        ]

        params_in = [
            0,
            1,
            1,
            1,
        ]

        params_mod = None

        x_in = [[1000, 1000, 1000], [1000, 1000, 1000]]

        expected_loss = 15.249237733900

        self._test_loss(y, x_in, params_mod, params_in, expected_loss)

    def test_train(self):
        model = LogisticRegression()

        x_train = [[1, 1], [-1, 1]]
        y_train = [1, 0]

        model.train(x_train, y_train)

        loss = model.loss(x_train, y_train)

        self.assertAlmostEqual(loss, 0.1, 1)

    def test_e2e(self):
        model = LogisticRegression()

        iris_dataset = load_iris()
        x_train = iris_dataset.data
        y_train = [0 if x == 0 else 1 for x in iris_dataset.target]

        conf = {
            "gamma": 1,
            "verbose": False,
            "n_iter": 1000,
        }

        loss_start = model.loss(x_train, y_train, np.zeros(x_train.shape[1] + 1))

        model.train(x_train, y_train, conf)
        loss_end = model.loss(x_train, y_train)

        loss_ratio = loss_end / loss_start

        self.assertLess(loss_ratio, 0.01)
