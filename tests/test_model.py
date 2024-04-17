from typing import List
import numpy as np
import pytest

import src.model as model

class LayerTest:
    def setup_method(self):
        self.layer : model.Layer = model.Layer(3, model.ReLU())

    def teardown_method(self):
        del self.layer

    def test_compute(self):
        self.layer.w = np.array([[1,2], [1, 1], [2, 3]])
        self.layer.b = np.array([1, 2, -10])
        a : np.ndarray = self.layer.compute([1, 1])
        assert a == np.array([4, 4, 0])

    def test_compute_2(self):
        self.layer.w = np.array([[1,2], [1, 1], [2, 3]])
        self.layer.b = np.array([1, 2, 3])
        a : np.ndarray = self.layer.compute([1, 1])
        assert a == np.array([4, 4, 8])

class ModelTest:
    def setup_method(self):
        self.layer1 = model.Layer(2, model.ReLU)
        self.layer2 = model.Layer(3, model.ReLU)
        self.layer3 = model.Layer(1, model.ReLU)

    def teardown_method(self):
        del self.layer1
        del self.layer2

    # The layer's output are
    # [1, 1] -> [4, 4] -> [13, 10, 23] -> [103]
    # based on this, all tests are made

    def test_compute(self):

        self.layer1.w = np.array([[1, 2], [1, 1]])
        self.layer1.b = np.array([1, 2])

        self.layer2.w = np.array([[1,2], [1, 1], [2, 3]])
        self.layer2.b = np.array([1, 2, 3])

        self.layer3.w = np.array([[1, 2, 3]])
        self.layer3.b = np.array([1])

        self.model = model.Model([self.layer1, self.layer2, self.layer3])

        a : np.ndarray = self.model.compute([1, 1])
        assert a == np.array([103])

    def test_compute(self):
        self.layer1.w = np.array([[1, 2], [1, 1]])
        self.layer1.b = np.array([1, 2])

        self.layer2.w = np.array([[1,2], [1, 1], [2, 3]])
        self.layer2.b = np.array([1, -11, -24])

        self.layer3.w = np.array([[1, 2, 3]])
        self.layer3.b = np.array([1])

        self.model = model.Model([self.layer1, self.layer2, self.layer3])

        a : np.ndarray = self.model.compute([1, 1])
        assert a == np.array([13])

def test_ReLU():
    assert model.ReLU(1) == 1
    assert model.ReLU(-1) == 0

def test_softmax():
    s = model.softmax(np.array([1, 2, 3]))
    assert np.isclose(s.sum(), 1)