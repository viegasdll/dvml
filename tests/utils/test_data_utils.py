import unittest

import numpy as np
from numpy.testing import assert_array_equal

from dvml.utils.data_utils import parse_x_lr, parse_x_ct


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
