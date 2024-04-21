import src.cost as cost
import numpy as np

import pytest

# There are two loss functions implemented
# Crossentropy loss

def test_cross_entropy_loss():
    y = np.array([0, 1, 0])
    a = np.array([0.1, 0.8, 0.1])
    expected = [-np.log(0.8)]
    assert np.array_equal(expected, cost.cross_entropy_loss(a, y))
