import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# Khai báo biến
x = sp.symbols('x')

# Tử và mẫu
numerator = sp.sqrt(x ** 2 + 3) - 2 * x
denominator = (x + 6) ** (1 / 3) + 2 * x - 1

# Biểu thức
expr = numerator / denominator

# Tính giới hạn khi x -> 2
limit_value = sp.limit(expr, x, 2)

# Hiển thị kết quả
print("Giá trị giới hạn khi x -> 2 là:", round(limit_value.evalf(), 2))

# Hàm số biểu diễn biểu thức trên


# Tạo mảng giá trị x trong đoạn [5, 30]
x_vals = np.linspace(5, 30, 400)
f_x = sp.lambdify(x, expr, 'numpy')
y_vals = f_x(x_vals)

# Vẽ đồ thị
plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_vals, label=r'$f(x) = \frac{\sqrt{x^2 + 3} - 2x}{\sqrt[3]{3x + 6} - 1}$', color='blue')
plt.title("Đồ thị hàm số f(x) trên đoạn [5, 30]")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)
plt.legend()
plt.show()
