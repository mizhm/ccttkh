import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

# Định nghĩa biến và hàm số
x = sp.symbols('x')
f_expr = sp.exp(-x/5) * sp.cos(2 * x)

# Tính đạo hàm của hàm f(x)
df_expr = sp.diff(f_expr, x)

# In ra đạo hàm
print(f"Đạo hàm của hàm số: {df_expr}")

# Chuyển hàm số và đạo hàm thành các hàm để vẽ đồ thị
f_lambdified = sp.lambdify(x, f_expr, 'numpy')
df_lambdified = sp.lambdify(x, df_expr, 'numpy')

# Tạo mảng x từ -10 đến 10
x_vals = np.linspace(-10, 10, 400)

# Tính giá trị của hàm số và đạo hàm tại các điểm x
y_vals = f_lambdified(x_vals)
dy_vals = df_lambdified(x_vals)

# Vẽ đồ thị hàm số y = f(x)
plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_vals, label=r'$y$', color='b')
plt.plot(x_vals, dy_vals, label=r"$y'$", color='r', linestyle='--')
plt.axhline(0, color='black',linewidth=1)
plt.axvline(0, color='black',linewidth=1)
plt.title("Đồ thị hàm số và đạo hàm")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()
