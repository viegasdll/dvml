import unittest

import numpy as np

from dvml.metrics.classification import accuracy


class TestAccuracy(unittest.TestCase):
    def test_acc_zero(self):
        y_real = [0, 0, 1, 1]
        y_pred = [1, 1, 0, 0]

        result = accuracy(y_real, y_pred)
        expected = 0

        self.assertEqual(result, expected)

    def test_acc_one(self):
        y_real = [0, 0, 1, 1]
        y_pred = [0, 0, 1, 1]

        result = accuracy(y_real, y_pred)
        expected = 1

        self.assertEqual(result, expected)

    def test_acc_some(self):
        y_real = [0, 0, 1, 1]
        y_pred = [1, 1, 1, 0]

        result = accuracy(y_real, y_pred)
        expected = 0.25

        self.assertEqual(result, expected)
