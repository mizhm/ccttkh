import numpy as np

# Ma trận hệ số A
A = np.array([
    [1, -1, 2],
    [2, 1, -1],
    [1, 1, 1]
])

# Vector kết quả a
a = np.array([4, -1, 5])

# Giải hệ phương trình Ax = a
solution = np.linalg.solve(A, a)

print("Nghiệm của hệ phương trình A là:")
print(f"x = {solution[0]:.4f}")
print(f"y = {solution[1]:.4f}")
print(f"z = {solution[2]:.4f}")

B = np.array([
    [5, -1, 2],
    [2, 2, -1],
    [1, 1, -1]
])

# Vector kết quả b
b = np.array([20, 23, 11])

# Giải hệ phương trình Bx = b
solution = np.linalg.solve(B, b)

print("Nghiệm của hệ phương trình B là:")
print(f"x = {solution[0]:.4f}")
print(f"y = {solution[1]:.4f}")
print(f"z = {solution[2]:.4f}")
