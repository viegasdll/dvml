import unittest

from numpy.testing import assert_array_almost_equal
import numpy as np
from dvml.utils.math_utils import (
    sigmoid,
    gini_binary,
    gini_binary_avg,
    gini_binary_split,
    gini_opt_split,
)


class TestParseXLR(unittest.TestCase):
    def test_empty_x(self):
        self.assertRaises(TypeError, sigmoid, None)

    def test_sigmoid(self):
        x_in = [
            0,
            -1000,
            1000,
        ]

        expected = [
            0.5,
            0,
            1,
        ]

        x_sigmoid = sigmoid(x_in)

        assert_array_almost_equal(x_sigmoid, expected)


class TestGiniBinary(unittest.TestCase):
    def test_zeros(self):
        y = np.zeros(100)

        result = gini_binary(y)
        expected = 0

        self.assertEqual(result, expected)

    def test_ones(self):
        y = np.ones(100)

        result = gini_binary(y)
        expected = 0

        self.assertEqual(result, expected)

    def test_half(self):
        y = np.concatenate([np.ones(100), np.zeros(100)], axis=0)

        result = gini_binary(y)
        expected = 0.5

        self.assertEqual(result, expected)

    def test_some(self):
        y = np.concatenate([np.ones(50), np.zeros(100)], axis=0)

        result = gini_binary(y)
        expected = 4 / 9

        self.assertAlmostEqual(result, expected, 3)


class TestGiniBinaryAvg(unittest.TestCase):
    def test_gini_binary_avg(self):
        y_left = np.concatenate([np.ones(100), np.zeros(100)], axis=0)
        y_right = np.ones(100)

        result = gini_binary_avg(y_left, y_right)
        expected = 1 / 3

        self.assertAlmostEqual(result, expected, 3)

    def test_left(self):
        y_left = np.concatenate([np.ones(100), np.zeros(100)], axis=0)
        y_right = []

        result = gini_binary_avg(y_left, y_right)
        expected = 0.5

        self.assertAlmostEqual(result, expected, 3)

    def test_right(self):
        y_left = []
        y_right = np.ones(100)

        result = gini_binary_avg(y_left, y_right)
        expected = 0

        self.assertAlmostEqual(result, expected, 3)


class TestGiniBinarySplit(unittest.TestCase):
    def test_gini_binary_split(self):
        x_in = [1, 2, 3, 4, 5, 6]
        y_in = [1, 1, 0, 0, 0, 0]
        boundary = 5

        result = gini_binary_split(x_in, y_in, boundary)
        expected = 1 / 3

        self.assertAlmostEqual(result, expected, 3)


class TestGiniOptSplit(unittest.TestCase):
    def test_gini_opt_split(self):
        x_in = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        y_in = [1, 0, 0, 0, 1, 1, 1, 1, 1]

        result = gini_opt_split(x_in, y_in)
        expected = (4.3, 1 / 6)

        assert_array_almost_equal(result, expected)

    def test_constant_x(self):
        x_in = [1, 1, 1, 1, 1, 1]
        y_in = [1, 0, 1, 0, 1, 0]

        result = gini_opt_split(x_in, y_in)
        expected = (1, 0.5)

        self.assertEqual(result, expected)
