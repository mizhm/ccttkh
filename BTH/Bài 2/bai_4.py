import numpy as np

def euclidean_distance(X: np.ndarray, Y: np.ndarray) -> float:
    difference = X - Y
    squared_difference = difference**2
    distance = np.sqrt(np.sum(squared_difference))
    return distance

def manhattan_distance(X: np.ndarray, Y: np.ndarray) -> float:
    difference = X - Y
    absolute_difference = np.abs(difference)
    distance = np.sum(absolute_difference)
    return distance

def minkowski_distance(X: np.ndarray, Y: np.ndarray, p: float) -> float:
    if p < 1:
        raise ValueError("Bậc 'p' của khoảng cách Minkowski phải là số dương (p >= 1).")
    absolute_difference = np.abs(X - Y)
    power_difference = absolute_difference**p
    distance = np.sum(power_difference)**(1/p)
    return distance

k = 4
x = np.random.rand(k)
y = np.random.rand(k)
print(x)
print(y)

print('Khoang cach euclidean: ', euclidean_distance(x, y))
print('Khoang cach manhattan: ', manhattan_distance(x, y))
print('Khoang cach minkowski: ', minkowski_distance(x, y, 3))
