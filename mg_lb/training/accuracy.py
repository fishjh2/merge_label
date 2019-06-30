import numpy as np


def max_first(array):
    return np.argmax(array, axis=1)


def max_forty(array):
    if array.shape[1] == 43:
        return np.argmax(array, axis=1)
    elif array.shape[2] == 43:
        return np.argmax(array, axis=2)


def max_multi(array):
    return np.argmax(array, axis=2)
