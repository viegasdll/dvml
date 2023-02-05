"""
Module for classification tree
"""
import random
from math import ceil, sqrt

import numpy as np

from dvml.models.model import SupervisedModel
from dvml.utils.config_utils import parse_config
from dvml.utils.math_utils import gini_opt_split


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

    def train(
        self, x_train, y_train, conf: dict = None
    ):  # pylint: disable=too-many-locals

        # Parse config
        parsed_config = parse_config(conf, self.DEFAULT_CONF)
        # Convert x_in to a numpy array
        x_form = np.array(x_train)
        # Make it into a matrix if it was just a vector
        if x_form.ndim == 1:
            x_form = x_form.reshape([1, len(x_form)])
        # Convert y to a numpy vector
        y_form = np.array(y_train)

        # Check if it's a leaf node. If so, compute a return value
        if parsed_config["leaf_node"]:
            self.return_val = np.mean(y_form)
            return 0

        # Check if y is all 1s or 0s. If so, compute a return value
        if np.sum(y_form) == 0 or np.sum(y_form) == len(y_form):
            self.return_val = y_form[0]
            return 0

        # Select the list of features to be tested
        feature_inds = list(range(len(x_form[0])))
        if parsed_config["n_features"] == "all":
            features = feature_inds
        elif parsed_config["n_features"] == "sqrt":
            n_feats = ceil(sqrt(len(feature_inds)))
            features = sorted(random.sample(feature_inds, n_feats))
        else:
            n_feats = max([int(parsed_config["n_features"]), len(feature_inds)])
            features = sorted(random.sample(feature_inds, n_feats))

        # For each feature in the list, compute a good boundary
        boundaries = [gini_opt_split(x_form[:, feat], y_form) for feat in features]
        # Find the best boundary
        opt_boundary, opt_gini = min(boundaries, key=lambda bdr: bdr[1])
        opt_feature = features[boundaries.index((opt_boundary, opt_gini))]

        # Overwrite the decision
        self.decision = {
            "feature": opt_feature,
            "boundary": opt_boundary,
        }

        # Split the dataset according to the decision
        x_left = x_form[x_form[:, opt_feature] < opt_boundary, :]
        y_left = y_form[x_form[:, opt_feature] < opt_boundary]
        x_right = x_form[x_form[:, opt_feature] > opt_boundary, :]
        y_right = y_form[x_form[:, opt_feature] > opt_boundary]

        return x_left, y_left, x_right, y_right


class ClassificationTreeModel(SupervisedModel):
    """
    Classification tree model class. Builds the model through
    individual nodes, depth-first
    """

    DEFAULT_CONF = {
        "n_features": "all",  # Can be either "all", "sqrt", or a number
    }

    def __init__(self):
        self.root_node = ClassificationTreeNode()

    def train(self, x_train, y_train, conf: dict = None):
        pass

    def predict(self, x_in):
        return self.root_node.predict(x_in)
