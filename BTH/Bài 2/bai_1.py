import numpy as np

k = int(input('Nhap k: '))
while k <= 0:
    print('k khong hop le')
    k = int(input('Nhap k: '))
A = np.random.rand(k, k)
B = np.random.rand(k, k)

#Phep cong
np.add(A, B)

#Phep tru
np.subtract(A, B)

#Phep nhan
np.dot(A, B)

#Phep nhan vo huong
A*B

# A^2
np.pow(A, 2)

#Ma tran A chuyen vi
np.transpose(A)

#Dinh thuc ma tran A
np.linalg.det(A)

#Dinh thuc ma tran B
np.linalg.det(B)
