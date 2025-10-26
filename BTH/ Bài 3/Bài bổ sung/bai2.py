import sympy as sp
import numpy as np

# Khai báo biến
x = sp.symbols('x')

# Biểu thức cần tích phân
expr = sp.sin(x) + 3 * sp.cos(x)
expr1 = (x ** 2 - 1) / (x ** 3 + x)

# Tính tích phân xác định từ 1 đến 3
integral = sp.integrate(expr, (x, 0, sp.pi / 2))
integral1 = sp.integrate(expr, (x, 1, 4))

# Hiển thị kết quả
print("Giá trị tích phân 1:", round(integral.evalf(), 2))
print("Giá trị tích phân 2:", round(integral1.evalf(), 2))
