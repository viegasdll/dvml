import unittest

import numpy as np

from dvml.utils.data_utils import parse_x_lr


class TestParseXLR(unittest.TestCase):
    def test_empty_x(self):
        self.assertRaises(TypeError, parse_x_lr, "asd")

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

        self.assertTrue(np.array_equal(expected, parse_x_lr(x_in)))
