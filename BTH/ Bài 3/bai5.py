import numpy as np
from scipy.optimize import fmin


# Định nghĩa hàm số
def f(x):
    return np.sin(x) ** 2 + x ** 4 + 100


res_min = fmin(f, 0)
res_max = fmin(lambda x: -f(x), 0)





