import unittest

import numpy as np
from numpy.testing import assert_array_equal

from dvml.models.classification_tree import (
    ClassificationTreeNode,
    ClassificationTreeModel,
)


class TestClassificationTreeNode(unittest.TestCase):
    def test_init(self):
        node = ClassificationTreeNode()

        self.assertTrue(node.left is None)
        self.assertTrue(node.right is None)
        self.assertEqual(node.return_val, 0.5)
        self.assertEqual(node.depth, 1)

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

    def test_train_leaf_node(self):
        node = ClassificationTreeNode()

        conf = {"leaf_node": True}

        x_train = np.array([1, 2, 3, 4, 5]).reshape([5, 1])
        y_train = [0, 1, 0, 1, 0]

        node.train(x_train, y_train, conf)

        print(x_train)

        result = node.predict(x_train)

        print(result)

        expected = 0.4 * np.ones(5)

        assert_array_equal(result, expected)

    def test_train_constant_y(self):
        node = ClassificationTreeNode()

        x_train = np.array([1, 2, 3, 4, 5]).reshape([5, 1])
        y_train = [1, 1, 1, 1, 1]

        node.train(x_train, y_train)

        result = node.predict(x_train)
        expected = np.ones(5)

        assert_array_equal(result, expected)

    def test_train_no_improve(self):
        node = ClassificationTreeNode()

        x_train = np.array([1, 1, 2, 2]).reshape([4, 1])
        y_train = [0, 1, 0, 1]

        node.train(x_train, y_train)

        result = node.predict(x_train)
        expected = 0.5 * np.ones(4)

        assert_array_equal(result, expected)

    def test_train_node(self):
        node = ClassificationTreeNode()

        x_train = np.array(
            [
                [1, 1, 1, 1, 1, 1, 1, 1],
                [0, 1, 0, 1, 0, 1, 0, 1],
                [-1, -1, -1, 3, 3, 3, 3, 3],
                [5, 5, 1, 1, 1, 1, 1, 1],
            ]
        ).transpose()
        y_train = [0, 0, 0, 0, 1, 1, 1, 1]

        x_left, y_left, x_right, y_right = node.train(x_train, y_train)

        x_left_expected = x_train[:3, :]
        x_right_expected = x_train[3:, :]

        self.assertEqual(node.decision["feature"], 2)
        self.assertEqual(node.decision["boundary"], -0.9)
        assert_array_equal(x_left, x_left_expected)
        assert_array_equal(x_right, x_right_expected)

    def test_train_node_sqrt(self):
        node = ClassificationTreeNode()

        x_train = np.array(
            [
                [1, 2, 3, 4, 5, 6, 7, 8],
                [0, 1, 0, 1, 0, 1, 0, 1],
                [-1, -1, -1, 3, 3, 3, 3, 3],
                [5, 5, 1, 1, 1, 1, 1, 1],
            ]
        ).transpose()
        y_train = [0, 0, 0, 0, 1, 1, 1, 1]

        conf = {"n_features": "sqrt"}

        node.train(x_train, y_train, conf)

        self.assertNotEqual(node.decision["boundary"], 0)

    def test_train_node_n_feats(self):
        node = ClassificationTreeNode()

        x_train = np.array(
            [
                [1, 2, 3, 4, 5, 6, 7, 8],
                [0, 1, 0, 1, 0, 1, 0, 1],
                [-1, -1, -1, 3, 3, 3, 3, 3],
                [5, 5, 1, 1, 1, 1, 1, 1],
            ]
        ).transpose()
        y_train = [0, 0, 0, 0, 1, 1, 1, 1]

        conf = {"n_features": 1}

        node.train(x_train, y_train, conf)

        self.assertNotEqual(node.decision["boundary"], 0)


class TestClassificationTreeModel(unittest.TestCase):
    def test_init(self):
        model = ClassificationTreeModel()

        self.assertTrue(model.root_node.left is None)
        self.assertTrue(model.root_node.right is None)
        self.assertEqual(model.root_node.return_val, 0.5)

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

        model = ClassificationTreeModel()

        model.root_node = node

        x_in = [
            [1, 2, 3],
            [0, -1, -1],
            [10, 15, 1],
        ]

        result = model.predict(x_in)
        expected = [0.8, 0.2, 0.8]

        assert_array_equal(result, expected)

    def test_predict_th(self):
        node = ClassificationTreeNode()

        left = ClassificationTreeNode(0.2)
        right = ClassificationTreeNode(0.8)

        node.left = left
        node.right = right

        node.decision = {
            "feature": 2,
            "boundary": 1,
        }

        model = ClassificationTreeModel()

        model.root_node = node

        x_in = [
            [1, 2, 3],
            [0, -1, -1],
            [10, 15, 1],
        ]

        result = model.predict_th(x_in)
        expected = [1, 0, 1]

        assert_array_equal(result, expected)

    def test_train(self):
        model = ClassificationTreeModel()

        x_train = np.array(
            [
                [1, 2, 3, 4, 5, 6, 7, 8],
                [0, 1, 0, 1, 0, 1, 0, 1],
                [-1, -1, -1, 3, 3, 3, 3, 3],
                [5, 5, 1, 1, 1, 1, 1, 1],
            ]
        ).transpose()
        y_train = [0, 0, 0, 0, 1, 1, 1, 1]

        model.train(x_train, y_train)

        result = model.predict(x_train)

        assert_array_equal(result, y_train)

    def test_train_max_depth_2(self):
        model = ClassificationTreeModel()

        x_train = np.array(
            [
                [1, 2, 3, 4, 5, 6, 7, 8],
                [0, 1, 0, 1, 0, 1, 0, 1],
                [-1, -1, -1, 3, 3, 3, 3, 3],
                [5, 5, 1, 1, 1, 1, 1, 1],
            ]
        ).transpose()
        y_train = [0, 0, 0, 0, 1, 1, 1, 1]

        conf = {
            "max_depth": 1,
        }

        model.train(x_train, y_train, conf)

        result = model.predict(x_train)
        expected = 0.5 * np.ones(8)

        assert_array_equal(result, expected)
