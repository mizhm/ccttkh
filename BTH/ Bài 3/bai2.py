import sympy as sp
import numpy as np

# Khai báo biến
x = sp.symbols('x')

# Biểu thức cần tích phân
expr = sp.exp(x) + 4
expr1 = 1/(x+1) - 1/(x+2)

# Tính tích phân xác định từ 1 đến 3
integral = sp.integrate(expr, (x, 1, 3))
integral1 = sp.integrate(expr, (x, 0, 1))

# Hiển thị kết quả
print("Giá trị tích phân 1:", integral.evalf())
print("Giá trị tích phân 2:", integral1.evalf())
