import math

n = int(input('n = '))
while n <= 0:
    print('n phai la so nguyen duong')
    n = int(input('n = '))
# cach 1
f1 = 1
for i in range(1, n + 1):
    f1 *= i

# cach 2
f2 = math.factorial(n)

print('f1 = ', f1)
print('f2 = ', f2)
