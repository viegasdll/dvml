"""
Module for random forest classifier
"""
import numpy as np

from dvml.models.classification_tree import ClassificationTreeModel
from dvml.models.model import SupervisedModel
from dvml.utils.config_utils import parse_config
from dvml.utils.data_utils import bootstrap, parse_x_ct


class RandomForestClassifier(SupervisedModel):
    """
    Random forest classifier, uses the ClassificationTreeModel
    class to build the decision trees
    """

    DEFAULT_CONF = {
        "n_features": "sqrt",  # Can be either "all", "sqrt", or a number
        "max_depth": 5,
        "n_trees": 32,
        "verbose": 1,
    }

    def __init__(self):
        self.trees = []

    def train(self, x_train, y_train, conf: dict = None):
        # Parse config
        parsed_config = parse_config(conf, self.DEFAULT_CONF)
        # Convert x_in to a numpy array
        x_form = parse_x_ct(x_train)
        # Convert y to a numpy vector
        y_form = np.array(y_train)

        # Repeat `n_trees` times:
        # 1. Create a bootstrap sample of the input
        # 2. Train a tree with that sample, and the input config
        if parsed_config["verbose"] > 0:
            print(f"Training random forest with {parsed_config['n_trees']} trees")

        for cur_tree in range(parsed_config["n_trees"]):
            if parsed_config["verbose"] > 0:
                print(f"Training tree {cur_tree}/{parsed_config['n_trees']}...")

            # Get a bootstrap sample
            x_sample, y_sample = bootstrap(x_form, y_form)

            # Train a new tree with the bootstrap sample
            new_tree = ClassificationTreeModel()
            new_tree.train(x_sample, y_sample, parsed_config)
            # Add it to the forest
            self.trees.append(new_tree)

        if parsed_config["verbose"] > 0:
            print("Training concluded")

    def predict(self, x_in):
        # If the list of trees is empty, no action
        if len(self.trees) < 1:
            raise RuntimeError("Model must be trained before predicting.")

        # Collect the prediction from each tree
        tree_predictions = [tree.predict(x_in) for tree in self.trees]

        # Return the average prediction
        return np.mean(tree_predictions, axis=0)

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
