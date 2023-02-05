import unittest

from dvml.models.classification_tree import ClassificationTreeNode


class TestClassificationTreeNode(unittest.TestCase):
    def test_init(self):
        node = ClassificationTreeNode()

        self.assertTrue(node.left is None)
        self.assertTrue(node.right is None)
        self.assertEqual(node.decision, 0.5)
