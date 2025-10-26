import numpy as np

# Khai báo ma trận A
A = np.array([
    [5.4, 3.1, 6.2, 1.5, 6.1, 3.2],
    [5.1, 8.5, 8.1, 3.5, 4.6, 2.2],
    [6.2, 2.1, 3.1, 3.3, 4.2, 5.8],
    [6.5, 3.5, 3.3, 3.3, 8.2, 4.8],
    [2.1, 4.6, 4.2, 8.2, 2.5, 1.6],
    [3.2, 8.2, 5.1, 4.8, 5.6, 8.2]
])

# Tính trị riêng và véc tơ riêng
eigenvalues, eigenvectors = np.linalg.eig(A)

# In kết quả
print("a. Các trị riêng (eigenvalues):")
print(np.round(eigenvalues, 2))

print("\nb. Các véc tơ riêng (eigenvectors) ứng với từng trị riêng (cột):")
print(np.round(eigenvectors, 2))
