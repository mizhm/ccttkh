import numpy as np

# Khai báo ma trận A
A = np.array([
    [3.2, 2.8, 4.1, 5.2, 1.9],
    [2.5, 4.3, 3.8, 5.6, 2.7],
    [4.8, 3.7, 5.5, 6.1, 3.9],
    [2.9, 3.4, 4.0, 5.7, 2.8],
    [3.1, 2.9, 3.6, 4.2, 2.5],
])

# Tính trị riêng và véc tơ riêng
eigenvalues, eigenvectors = np.linalg.eig(A)

# In kết quả
print("a. Các trị riêng (eigenvalues):")
print(np.round(eigenvalues, 2))

print("\nb. Các véc tơ riêng (eigenvectors) ứng với từng trị riêng (cột):")
print(np.round(eigenvectors, 2))
