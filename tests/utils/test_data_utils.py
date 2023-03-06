import unittest

import numpy as np
from numpy.testing import assert_array_equal

from dvml.utils.data_utils import parse_x_lr, parse_x_ct, bootstrap


class TestParseXLR(unittest.TestCase):
    def test_empty_x(self):
        self.assertRaises(TypeError, parse_x_lr, None)

    def test_x(self):
        x_in = [
            [1, 3],
            [0, 0],
        ]

        expected = np.array(
            [
                [1, 1, 3],
                [1, 0, 0],
            ]
        )

        assert_array_equal(expected, parse_x_lr(x_in))


class TestParseXCT(unittest.TestCase):
    def test_x(self):
        x_in = [
            [1, 3],
            [0, 0],
        ]

        expected = np.array(
            [
                [1, 3],
                [0, 0],
            ]
        )

        assert_array_equal(expected, parse_x_ct(x_in))

    def test_x_1d(self):
        x_in = [1, 2, 3]
        result = parse_x_ct(x_in)

        expected = np.array([1, 2, 3]).reshape([1, 3])

        assert_array_equal(expected, result)
        assert_array_equal(result.shape, expected.shape)


class TestBootstrap(unittest.TestCase):
    def test_full(self):
        x_in = np.array(
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
                [0, 0, 0],
            ]
        )

        y_in = np.array([1, 2, 3, 4])

        x_out, y_out = bootstrap(x_in, y_in)

        expected_shape = (4, 3)

        self.assertEqual(x_out.shape, expected_shape)

    def test_nsamples(self):
        x_in = np.array(
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
                [0, 0, 0],
            ]
        )

        y_in = np.array([1, 2, 3, 4])
        n_samples = 2

        x_out, y_out = bootstrap(x_in, y_in, n_samples)

        expected_shape = (n_samples, 3)

        self.assertEqual(x_out.shape, expected_shape)
