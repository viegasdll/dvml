import unittest

from sklearn.datasets import load_breast_cancer

from dvml.metrics.classification import accuracy
from dvml.models.random_forest import RandomForestClassifier


class TestRandomForest(unittest.TestCase):
    def test_e2e(self):
        model = RandomForestClassifier()

        bc_dataset = load_breast_cancer()
        x_train = bc_dataset.data
        y_train = bc_dataset.target

        model.train(x_train, y_train)

        y_pred = model.predict_th(x_train)

        acc_pred = accuracy(y_train, y_pred)

        self.assertGreater(acc_pred, 0.9)

    def test_e2e_silent(self):
        model = RandomForestClassifier()

        bc_dataset = load_breast_cancer()
        x_train = bc_dataset.data
        y_train = bc_dataset.target

        model_conf = {"verbose": 0}

        model.train(x_train, y_train, model_conf)

        y_pred = model.predict_th(x_train)

        acc_pred = accuracy(y_train, y_pred)

        self.assertGreater(acc_pred, 0.9)

    def test_predict_error(self):
        model = RandomForestClassifier()

        self.assertRaises(RuntimeError, model.predict, None)
