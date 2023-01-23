import numpy as np

from dvml.models.linear import LinearRegression
import pytest


def test_set_params():
    model = LinearRegression()
    test_params = [0.1, 0.2, 0.3]

    model.set_params(test_params)

    assert np.array_equal(model.params, test_params)
