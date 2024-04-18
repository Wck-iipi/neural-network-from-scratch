import numpy as np

def cross_entropy_loss(a:np.ndarray, y:np.ndarray):
    return -np.log(a[y])

def difference(a:np.ndarray, y:np.ndarray):
    return y - a
