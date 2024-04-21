from collections.abc import Callable
from typing import List
import numpy as np

def ReLU(z: np.ndarray) -> np.ndarray:
    return np.array([np.fromiter([np.max([0, z_i]) for z_i in z[0]], float)])

def ReLU_prime(z: np.ndarray) -> np.ndarray:
    return np.array([np.fromiter([0 if z_i <= 0 else 1 for z_i in z[0]], float)])

def softmax(z: np.ndarray) -> np.ndarray:
    sum = np.sum(np.fromiter([np.exp(z_i) for z_i in z[0]], float))
    return np.array([np.fromiter([np.exp(z_i) / sum for z_i in z[0]], float)])


class Layer:
    def __init__(self, n: int, f: Callable[[np.ndarray], np.ndarray], f_prime: Callable[[np.ndarray], np.ndarray]):
        self.n = n
        self.f = f
        self.f_prime = f_prime
        self.b = np.zeros(n)
        self.w: np.ndarray = np.array([])
        self.a: np.ndarray = np.array([])
        self.z: np.ndarray = np.array([])

    def setWeights(self, m: int):
        self.w = np.zeros((self.n, m))

    def compute(self, x: np.ndarray):
        if self.w.size == 0:
            self.setWeights(x.shape[0])

        for i in range(self.n):
            print("w[i]: ", self.w[i])
            print("np.transpose(x): ", np.transpose(x))
            print("np.dot(self.w[i], np.transpose(x)): ", np.dot(self.w[i], np.transpose(x)))
            print("self.b[i]: ", self.b[i])
            print("np.dot(self.w[i], np.transpose(x)) + self.b[i]: ", np.dot(self.w[i], np.transpose(x)) + self.b[i])
        self.z = np.array([np.fromiter([np.dot(self.w[i], np.transpose(x)) + self.b[i] for i in range(self.n)], float)])
        print("z: ", self.z)
        self.a = self.f(self.z)
        return self.a

class Model:
    def __init__(self, layers: List[Layer]):
        self.layers = layers

    def compute(self, x: np.ndarray):
        for layer in self.layers:
            x = layer.compute(x)
        return x
