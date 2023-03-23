"""
Module for neural network (mlp)
"""
import numpy as np

from dvml.models.model import Model
from dvml.utils.config_utils import parse_config
from dvml.utils.data_utils import parse_x_ct


def relu(x_in):
    """

    :param x_in:
    :return:
    """
    return max(x_in, 0)


class Neuron(Model):
    """
    Class for single node of a neural network
    """

    DEFAULT_CONF = {
        "activation": "relu",
    }

    def __init__(
        self, nd_in: int = None, weights: np.ndarray = None, conf: dict = None
    ):
        # Parse config
        self.parsed_config = parse_config(conf, self.DEFAULT_CONF)

        if self.parsed_config["activation"] == "relu":
            self.activation = relu
        else:
            raise ValueError("Invalid activation function!")

        if weights is None:
            self.nd_in = nd_in
            self.weights = np.random.random(nd_in)
        else:
            self.nd_in = len(weights)
            self.weights = np.array(weights)

    def predict(self, x_in):
        # Convert x_in to a numpy array
        x_form = parse_x_ct(x_in)

        # Compute the prediction for each row of the dataset, and collect them into a vector
        return np.apply_along_axis(self.predict_one, axis=1, arr=x_form)

    def predict_one(self, x_vec):
        """

        :param x_vec: 1-D array of features
        :return: prediction for the input features vector
        """
        x_w = np.dot(x_vec, self.weights)

        return self.activation(x_w)


class Layer(Model):
    """
    Class for a single layer of a neural network
    """

    DEFAULT_CONF = {
        "activation": "relu",
        "weights_init": "random",
    }

    def __init__(self, n_nodes: int, nd_in: int, conf: dict = None):
        # Parse config
        self.parsed_config = parse_config(conf, self.DEFAULT_CONF)

        self.nd_in = nd_in
        self.n_nodes = n_nodes

        if self.parsed_config["weights_init"] == "random":
            self.weights_init = "random"
        else:
            raise ValueError("Invalid weights initialization option!")

        if self.parsed_config["activation"] == "relu":
            self.activation = relu
        else:
            raise ValueError("Invalid activation function!")

        # Initialize the nodes - add conditions once we have different initializations
        self.neurons = [
            Neuron(nd_in=self.nd_in, conf=self.parsed_config)
            for _ in range(self.n_nodes)
        ]

    def predict(self, x_in):
        # Convert x_in to a numpy array
        x_form = parse_x_ct(x_in)

        # Compute the prediction for each row of the dataset, and collect them into a vector
        return np.apply_along_axis(self.predict_one, axis=1, arr=x_form)

    def predict_one(self, x_vec):
        """

        :param x_vec: 1-D array of features
        :return: prediction for the input features vector
        """
        return np.array([neuron.predict_one(x_vec) for neuron in self.neurons])
