from src import cost
import src.backprop as backprop

import numpy as np
import src.model as model

def test_backtracking():
    layer1 = model.Layer(2, model.ReLU)
    layer1.w = np.array([[0.11, 0.12], [0.21, 0.03]])
    layer1.b = np.array([0, 0])

    layer2 = model.Layer(1, model.ReLU)
    layer2.w = np.array([[0.14], [0.15]])
    layer2.b = np.array([0])

    model_1 = model.Model([layer1, layer2])

    model_first_iteration = backprop.backprop(model_1, np.array([2, 3]), cost.difference)
    assert np.array_equal(model_first_iteration.layers[0].w, np.array([[0.12, 0.13], [0.23, 0.1]]))
    assert np.array_equal(model_first_iteration.layers[1].w, np.array([[0.17, 0.17]]))

    model_second_iteration = backprop.backprop(model_1, np.array([2, 3]), cost.difference)
    assert np.array_equal(model_second_iteration.layers[0].w, np.array([[0.13, 0.14], [0.25, 0.12]]))
    assert np.array_equal(model_second_iteration.layers[1].w, np.array([[0.2, 0.19]]))
