import numpy as np
from scipy.optimize import fmin


# Định nghĩa hàm số
def f(x):
    return x ** 3 - 6 * (x ** 2) + 9 * x + 2


res_min = fmin(f, 0)
res_max = fmin(lambda x: -f(x), 0)
