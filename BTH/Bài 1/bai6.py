import math

n = int(input('n = '))
while n <= 0:
    print('n phai la so nguyen duong')
    n = int(input('n = '))
x = float(input('x = '))

f = 0
sum = 0
if n < 20:
    for i in range(1, n + 1):
        sum += i
        f += (x - i) / sum
else:
    f = math.sqrt(n) + x

print("F = ", f)
