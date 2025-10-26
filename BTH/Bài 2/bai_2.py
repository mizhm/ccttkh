import numpy as np

n = int(input("Nhap so hang ma tran A: "))
m = int(input("Nhap so cot ma tran A: "))
A = np.random.rand(n,m)
A

#Tinh toan tren cot
print("Cac phep toan tren cot")
print('Sum: ', np.sum(A, axis=0))
print('Min: ', np.min(A, axis=0))
print('Max: ', np.max(A, axis=0))
print('Mean: ', np.mean(A, axis=0))

#Tinh toan tren hang
print("Cac phep toan tren hang")
print('Sum: ', np.sum(A, axis=1))
print('Min: ', np.min(A, axis=1))
print('Max: ', np.max(A, axis=1))
print('Mean: ', np.mean(A, axis=1))

#Tinh toan tren toan ma tran
print("Phep toan tren toan ma tran")
print('Sum: ', np.sum(A))
print('Minj: ', np.min(A))
print('Max: ', np.max(A))
print('Mean: ', np.mean(A))
