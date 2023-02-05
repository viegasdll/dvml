"""
Module for classification tree
"""
from dvml.models.model import SupervisedModel


class ClassificationTreeNode(SupervisedModel):
    """
    Individual node for a classification tree
    """

    def __init__(self):
        # Define the left and right successors, set to empty to start with
        self.left = None
        self.right = None
        # Set a default value, in case it becomes a leaf node
        # The training process will either define successors, or update the decision
        self.decision = 0.5

    def predict(self, x_in):
        pass

    def train(self, x_train, y_train, conf: dict = None):
        pass
