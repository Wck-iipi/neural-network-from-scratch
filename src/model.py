from collections.abc import Callable
from typing import List
import numpy as np

def ReLU(x):
    if x < 0:
        return 0
    else:
        return x


class Layer:
    def __init__(self, n: int, f: Callable[[float], float]):
        self.n = n
        self.f = f
        self.b = np.zeros(n)
        self.w: np.ndarray = np.array([])

    def setWeights(self, m: int):
        self.w = np.zeros((self.n, m))

    def compute(self, x: np.ndarray):
        if self.w.size == 0:
            self.setWeights(x.shape[0])

        a: np.ndarray = np.zeros(self.n)
        for r in range(self.n):
            a[r] = self.f(np.dot(self.w[r], x) + self.b[r])
        return a

class Model:
    def __init__(self, layers: List[Layer]):
        self.layers = layers

    def compute(self, x: np.ndarray):
        for layer in self.layers:
            x = layer.compute(x)
        return x
