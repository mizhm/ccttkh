import math

r = float(input('r = '))
while r <= 0:
    print('r phai la so duong!')
    r = float(input('r = '))

a = float(input('a = '))
b = float(input('b = '))
x = float(input('x = '))
y = float(input('y = '))

if math.sqrt((x - a) ** 2 + (y - b) ** 2) <= r:
    print(True)
else:
    print(False)
