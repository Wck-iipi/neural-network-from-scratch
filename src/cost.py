import numpy as np

def cross_entropy_loss(a:np.ndarray, y:int):
    return -np.log(a[y])
