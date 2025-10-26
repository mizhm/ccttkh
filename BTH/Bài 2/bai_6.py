import numpy as np

n = int(input("Nhap so hang ma tran A: "))
m = int(input("Nhap so cot ma tran A: "))
b = float(input("b: "))
A = np.random.rand(n,m)
A

print("Phep cong: \n", A + b)
print("Phep tru: \n", A - b)
print("Phep nhan: \n", A * b)
print("Phep chia: \n", A / b)
print("Phep lay du: \n", A % b)
