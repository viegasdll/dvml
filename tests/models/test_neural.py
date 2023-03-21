import unittest

import numpy as np

from dvml.models.neural import relu, Neuron
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
