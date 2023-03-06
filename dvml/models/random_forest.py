"""
Module for random forest classifier
"""

from dvml.models.model import SupervisedModel


class RandomForestClassifier(SupervisedModel):
    """
    Random forest classifier, uses the ClassificationTreeModel
    class to build the decision trees
    """

    def train(self, x_train, y_train, conf: dict = None):
        pass

    def predict(self, x_in):
        pass
