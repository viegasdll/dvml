"""
Module for random forest classifier
"""
import numpy as np

from dvml.models.model import SupervisedModel


class RandomForestClassifier(SupervisedModel):
    """
    Random forest classifier, uses the ClassificationTreeModel
    class to build the decision trees
    """

    DEFAULT_CONF = {
        "n_features": "all",  # Can be either "all", "sqrt", or a number
        "max_depth": None,
        "n_trees": 32,
    }

    def __init__(self):
        self.trees = []

    def train(self, x_train, y_train, conf: dict = None):
        pass

    def predict(self, x_in):
        # If the list of trees is empty, no action
        if len(self.trees) < 1:
            raise RuntimeError("Model must be trained before predicting.")

        # Collect the prediction from each tree
        tree_predictions = [tree.predict(x_in) for tree in self.trees]

        # Return the average prediction
        return np.mean(tree_predictions)

    def predict_th(self, x_in, threshold=0.5):
        """
        Predicts then applies a threshold

        :param x_in: numpy matrix (or equivalent) of features
        :param threshold: classification threshold. Default=0.5
        :return: 1-D array of binary predictions
        """
        return np.array(
            [0 if y_pred < threshold else 1 for y_pred in self.predict(x_in)]
        )
