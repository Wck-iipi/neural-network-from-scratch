import numpy as np

def cross_entropy_loss(a:np.ndarray, y:np.ndarray):
    for r in range(len(y)):
        if y[r] == 1:
            return -np.array([np.log(a[r])])
    return -np.array([-1])

def difference(a:np.ndarray, y:np.ndarray):
    return y - a

def difference_prime(a:np.ndarray, y:np.ndarray):
    return np.array([1 for _ in range(len(a))])
