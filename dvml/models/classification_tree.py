"""
Module for classification tree
"""
import numpy as np

from dvml.models.model import SupervisedModel


class ClassificationTreeNode(SupervisedModel):
    """
    Individual node for a classification tree
    """

    DEFAULT_CONF = {
        "n_features": "all",  # Can be either "all", "sqrt", or a number
        "leaf_node": False,
    }

    def __init__(self, return_val=0.5):
        # Define the left and right successors, set to empty to start with
        self.left = None
        self.right = None
        # Set a default value, in case it becomes a leaf node
        # The training process will either define successors, or update the return value
        self.return_val = return_val
        # Initialize a dummy decision function
        self.decision = {
            "feature": 0,  # The index of the feature used in the decision
            "boundary": 0,  # The boundary value that decides between left/right successor
        }

    def predict(self, x_in):
        # Convert x_in to a numpy array
        x_form = np.array(x_in)
        # Make it into a matrix if it was just a vector
        if x_form.ndim == 1:
            x_form = x_form.reshape([1, len(x_form)])

        # Compute the prediction for each row of the dataset, and collect them into a vector
        return np.apply_along_axis(self.predict_one, axis=1, arr=x_form)

    def predict_one(self, x_vec):
        """

        :param x_vec:
        :return:
        """
        # If either successor is not defined, return the existing return value
        if self.left is None or self.right is None:
            return self.return_val
        # Otherwise, apply the decision function and collect the return value from the successor
        if x_vec[self.decision["feature"]] < self.decision["boundary"]:
            return self.left.predict_one(x_vec)
        return self.right.predict_one(x_vec)

    def train(self, x_train, y_train, conf: dict = None):
        # Overall scheme:
        # If leaf_node = True, compute the decision based on y_train
        # If y_train is all 1 or 0, set the decision to that value
        # Otherwise, define the boundary
        # 1. select a subset of feature indices based on n_feats
        # 2. for each feature, determine the optimal boundary value
        # 3. pick the best, overwrite the decision object
        # Note: need the gini function, for both optimization and tests
        pass
