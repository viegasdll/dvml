import unittest

import numpy as np

from dvml.models.classification_tree import ClassificationTreeNode
from numpy.testing import assert_array_equal


class TestClassificationTreeNode(unittest.TestCase):
    def test_init(self):
        node = ClassificationTreeNode()

        self.assertTrue(node.left is None)
        self.assertTrue(node.right is None)
        self.assertEqual(node.return_val, 0.5)

    def test_predict_one_01(self):
        node = ClassificationTreeNode(return_val=0.2)
        x_vec = np.array([1, 2, 3])

        expected = 0.2

        self.assertEqual(node.predict_one(x_vec), expected)

    def test_predict_one_02(self):
        node = ClassificationTreeNode()

        left = ClassificationTreeNode(0.2)
        right = ClassificationTreeNode(0.8)

        node.left = left
        node.right = right

        x_vec = np.array([1, 2, 3])

        node.decision = {
            "feature": 1,
            "boundary": 4,
        }

        expected = 0.2

        self.assertEqual(node.predict_one(x_vec), expected)

    def test_predict_one_03(self):
        node = ClassificationTreeNode()

        left = ClassificationTreeNode(0.2)
        right = ClassificationTreeNode(0.8)

        node.left = left
        node.right = right

        x_vec = np.array([1, 2, 3])

        node.decision = {
            "feature": 2,
            "boundary": 1,
        }

        expected = 0.8

        self.assertEqual(node.predict_one(x_vec), expected)

    def test_predict_single_vec(self):
        node = ClassificationTreeNode(return_val=0.2)
        x_vec = np.array([1, 2, 3])

        expected = [0.2]

        assert_array_equal(node.predict(x_vec), expected)

    def test_predict_matrix(self):
        node = ClassificationTreeNode()

        left = ClassificationTreeNode(0.2)
        right = ClassificationTreeNode(0.8)

        node.left = left
        node.right = right

        node.decision = {
            "feature": 2,
            "boundary": 1,
        }

        x_in = [
            [1, 2, 3],
            [0, -1, -1],
            [10, 15, 1],
        ]

        result = node.predict(x_in)
        expected = [0.8, 0.2, 0.8]

        assert_array_equal(result, expected)
