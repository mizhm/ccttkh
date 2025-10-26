import numpy as np
from random import randint

def my_func(A):
    max = np.max(A, axis = 1)
    min = np.min(A, axis = 1)
    return np.sum(max - min)

k = randint(1, 10)
A = np.random.rand(k, k)
my_func(A)
