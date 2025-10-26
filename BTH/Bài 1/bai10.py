import math

r = float(input('r = '))
while r <= 0:
    print('r phai la so duong!')
    r = float(input('r = '))

print('Dien tich hinh cau: ', 4 * math.pi * (r ** 2))
print('The tich hinh cau: ', (4 / 3) * math.pi * (r ** 3))
