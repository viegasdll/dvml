import unittest

import numpy as np

from dvml.models.neural import relu, Neuron, Layer
from numpy.testing import assert_array_equal


class TestRelu(unittest.TestCase):
    def test_pos(self):
        expected = 42
        result = relu(42)

        self.assertEqual(expected, result)

    def test_neg(self):
        expected = 0
        result = relu(-10)

        self.assertEqual(expected, result)


class TestNeuron(unittest.TestCase):
    def test_init_dim(self):
        neuron = Neuron(nd_in=5)

        result = len(neuron.weights)
        expected = 5

        self.assertEqual(expected, result)
        self.assertEqual(expected, neuron.nd_in)

    def test_init_weights(self):
        weights = np.array([1, 2, 3, 4, 5])

        neuron = Neuron(weights=weights)

        self.assertEqual(neuron.nd_in, 5)
        assert_array_equal(weights, neuron.weights)

    def test_predict_one(self):
        weights = np.array([1, 2, 3])

        neuron = Neuron(weights=weights)

        result = neuron.predict([1, -1, 1])
        expected = 2

        self.assertEqual(expected, result)

    def test_predict_mult(self):
        weights = np.array([1, 2, 3])

        neuron = Neuron(weights=weights)

        x_in = [
            [1, -1, 1],
            [-4, -1, -2],
        ]

        result = neuron.predict(x_in)
        expected = [2, 0]

        assert_array_equal(expected, result)

    def test_wrong_activation(self):
        neuron_conf = {
            "activation": "what is this?",
        }

        self.assertRaises(ValueError, Neuron, conf=neuron_conf)


class TestLayer(unittest.TestCase):
    def test_init(self):
        layer = Layer(12, 5)

        self.assertEqual(len(layer.neurons), 12)
        self.assertEqual(len(layer.neurons[0].weights), 5)

    def test_wrong_activation(self):
        layer_conf = {
            "activation": "what is this?",
        }

        self.assertRaises(ValueError, Layer, n_nodes=3, nd_in=3, conf=layer_conf)

    def test_wrong_weights(self):
        layer_conf = {
            "weights_init": "what is this?",
        }

        self.assertRaises(ValueError, Layer, n_nodes=3, nd_in=3, conf=layer_conf)

    def test_predict_one(self):
        layer = Layer(12, 3)

        x_in = np.array([1, 2, 3])

        result = layer.predict_one(x_in)

        # Check output size
        self.assertEqual(result.shape, (12,))
        # Check that the random init yields distinct weights
        self.assertNotEqual(result[0], result[3])

    def test_predict(self):
        layer = Layer(12, 3)

        x_in = [
            [1, -1, 1],
            [-4, -1, -2],
            [3, 3, 3],
            [1, 0.5, 1],
        ]

        result = layer.predict(x_in)

        # Check that we get the expected number of output vectors
        self.assertEqual(result.shape, (4, 12))
        # Check that we don't get the same output in different neurons with random init
        self.assertNotEqual(result[0][2], result[3][2])
