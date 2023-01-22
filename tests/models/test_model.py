from dvml.models.model import ConstantModel
import numpy as np
import pytest

@pytest.mark.parametrize(
    'y, x_in, x_out',
    [
        (.5, np.ones(10), .5 * np.ones(10)),
        (.1, np.linspace(1, 10, 50), .1 * np.ones(50)),
        (.4, np.random.random(100), .4 * np.ones(100)),
    ]
)
def test_constant_model(y, x_in, x_out):
    model = ConstantModel(y)

    assert np.array_equal(model.predict(x_in), x_out)
