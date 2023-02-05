import unittest

from numpy.testing import assert_array_almost_equal
import numpy as np
from dvml.utils.math_utils import sigmoid, gini_binary


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
