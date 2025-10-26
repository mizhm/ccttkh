import math

r = float(input("Nhap vao ban kinh r = "))
while r <= 0:
    print("Ban kinh phai la so duong!")
    r = float(input("Nhap vao ban kinh r = "))
print(f"Dien tich hinh tron: {r ** 2 * math.pi:.2f}")
