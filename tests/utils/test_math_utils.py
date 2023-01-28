import unittest

from numpy.testing import assert_array_almost_equal

from dvml.utils.math_utils import sigmoid


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
