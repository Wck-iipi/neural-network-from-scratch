from collections.abc import Callable
import numpy as np

from src.model import Model

# We have the following equations:
# dL/dW = (a^(L-1))^T  * dL/da^(L) * diagonal(activation_prime(z^(L)))
# dL/db = dL/da^(L) * diagonal(activation_prime(z^(L)))
# dL/da^(L-1) = W^T * dL/da^(L) * diagonal(activation_prime(z^(L)))

def calculate_dl_dW(a_previous: np.ndarray, delta: np.ndarray, z: np.ndarray, activation_prime: Callable[[np.ndarray], np.ndarray]):
    return np.dot(np.transpose(a_previous), np.dot(delta, np.diag(activation_prime(z))))

def calculate_dl_db(delta: np.ndarray, z: np.ndarray, activation_prime: Callable[[np.ndarray], np.ndarray]):
    return np.dot(delta, np.diag(activation_prime(z)))

def calculate_dl_da_previous(W: np.ndarray, delta: np.ndarray, z:np.ndarray, activation_prime: Callable[[np.ndarray], np.ndarray]):
    return np.dot(W, np.dot(delta, np.diag(activation_prime(z))))

def backpropagation(model: Model, loss_function_prime: Callable[[np.ndarray, np.ndarray], np.ndarray], Y: np.ndarray):
    # Assuming forward propagation has been done
    dl_da = loss_function_prime(model.layers[-1].a, Y)
    print(dl_da)

    dl_db = np.array([calculate_dl_db(dl_da, model.layers[-1].z, model.layers[-1].f_prime)])
    dl_dW = np.array([calculate_dl_dW(model.layers[-2].a, dl_da, model.layers[-1].z, model.layers[-1].f)])

    for i in range(len(model.layers) - 2, 0, -1):
        dl_da = calculate_dl_da_previous(model.layers[i].w, dl_da, model.layers[i].z, model.layers[i].f_prime)
        np.append(dl_db, calculate_dl_db(dl_da, model.layers[i].z, model.layers[i].f_prime))
        np.append(dl_dW, calculate_dl_dW(model.layers[i-1].a, dl_da, model.layers[i].z, model.layers[i].f))

    return dl_dW, dl_db
