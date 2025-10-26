import math

h = float(input('Nhap do cao h: '))
while h <= 0:
    print('h phai la so duong')
    h = float(input('Nhap do cao h: '))
print(f"van toc v: {math.sqrt(2 * 9.8 * h):.2f}")
