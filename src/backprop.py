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
    return np.dot(np.transpose(W), np.dot(delta, np.diag(activation_prime(z))))

def backpropagation(model: Model, loss_function_prime: Callable[[np.ndarray, np.ndarray], np.ndarray], Y: np.ndarray):
    # Assuming forward propagation has been done
    dl_da = loss_function_prime(model.layers[-1].a, Y)

    dl_db = calculate_dl_db(dl_da, model.layers[-1].z, model.layers[-1].f_prime)
    dl_dW = calculate_dl_dW(model.layers[-2].a, dl_da, model.layers[-1].z, model.layers[-1].f_prime)

    dl_db_collection = np.array([dl_db])
    dl_dW_collection = np.array([dl_dW])
    for i in range(len(model.layers) - 2, -1, -1):
        dl_da = calculate_dl_da_previous(model.layers[i+1].w, dl_da, model.layers[i+1].z, model.layers[i+1].f_prime)
        dl_db = calculate_dl_db(dl_da, model.layers[i].z, model.layers[i].f_prime)
        dl_dW = calculate_dl_dW(model.layers[i].a, dl_da, model.layers[i].z, model.layers[i].f_prime)
        np.append(dl_db_collection, dl_db)
        np.append(dl_dW_collection, dl_dW)

    return dl_dW_collection, dl_db_collection
