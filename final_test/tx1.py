import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.linalg import det, inv, eig

# Định nghĩa ma trận A (6x6)
A = np.array([
    [-1.4, 1.1, 2.2, -1.5, 2.1, 3.2],
    [1.1, 1.5, -2.1, 3.5, 4.6, 2.2],
    [-2.2, 2.1, 1.1, 3.3, 4.2, 2.8],
    [1.5, -3.5, 3.3, 1.3, 1.2, -4.8],
    [2.1, 4.6, -4.2, 1.2, -2.5, 1.6],
    [3.2, 2.2, 2.8, 4.8, 1.6, 1.2]
])

print("=" * 60)
print("BÀI 1: GIẢI TOÁN MA TRẬN")
print("=" * 60)

print("\nMa trận A:")
print(A)

# a. Tìm định thức |A| và ma trận nghịch đảo của A
print("\n" + "-" * 60)
print("a. Tìm |A| và ma trận nghịch đảo của A")
print("-" * 60)

det_A = det(A)
print(f"\nĐịnh thức |A| = {det_A:.6f}")

inv_A = inv(A)
print("\nMa trận nghịch đảo A^(-1):")
print(np.round(inv_A, 6))

# Kiểm tra: A * A^(-1) = I
print("\nKiểm tra: A * A^(-1) ≈ I:")
print(np.round(A @ inv_A, 6))

# b. Tìm C = Tổng max của các hàng
print("\n" + "-" * 60)
print("b. Tìm C = Tổng max của các hàng")
print("-" * 60)

max_each_row = np.max(A, axis=1)
print("\nGiá trị max của mỗi hàng:")
for i, max_val in enumerate(max_each_row):
    print(f"  Hàng {i + 1}: max = {max_val}")

C = np.sum(max_each_row)
print(f"\nC = Tổng max các hàng = {C}")

# c. Tính các trị riêng và véc tơ riêng tương ứng của A
print("\n" + "-" * 60)
print("c. Tính các trị riêng và véc tơ riêng tương ứng của A")
print("-" * 60)

eigenvalues, eigenvectors = eig(A)
print("\nCác trị riêng (eigenvalues):")
for i, val in enumerate(eigenvalues):
    if np.iscomplex(val):
        print(f"  λ{i + 1} = {val}")
    else:
        print(f"  λ{i + 1} = {val.real:.6f}")

print("\nCác véc tơ riêng (eigenvectors) tương ứng (theo cột):")
print(np.round(eigenvectors, 6))

# Kiểm tra: A * v = λ * v
print("\nKiểm tra A * v ≈ λ * v (với véc tơ riêng thứ nhất):")
v1 = eigenvectors[:, 0]
lambda1 = eigenvalues[0]
print(f"  A * v1 = {np.round(A @ v1, 6)}")
print(f"  λ1 * v1 = {np.round(lambda1 * v1, 6)}")

# d. Giải hệ phương trình: 2x - 3y = 1; x + 4y = 6
print("\n" + "-" * 60)
print("d. Giải hệ phương trình:")
print("   2x - 3y = 1")
print("   x + 4y = 6")
print("-" * 60)

# Hệ phương trình dạng Ax = b
coeff_matrix = np.array([
    [2, -3],
    [1, 4]
])
b_vector = np.array([1, 6])

solution = np.linalg.solve(coeff_matrix, b_vector)
x, y = solution
print(f"\nNghiệm của hệ:")
print(f"  x = {x:.6f}")
print(f"  y = {y:.6f}")

# Kiểm tra
print("\nKiểm tra:")
print(f"  2*{x:.6f} - 3*{y:.6f} = {2 * x - 3 * y:.6f} (phải = 1)")
print(f"  {x:.6f} + 4*{y:.6f} = {x + 4 * y:.6f} (phải = 6)")

# e. Tính đạo hàm và vẽ đồ thị của f(x) = x³ - 3x² + 6x - 10 với x ∈ [-100, 100]
print("\n" + "-" * 60)
print("e. Tính đạo hàm và vẽ đồ thị của f(x) = x³ - 3x² + 6x - 10")
print("-" * 60)

# Sử dụng sympy để tính đạo hàm
import sympy as sp

# Định nghĩa biến symbolic
x_sym = sp.Symbol('x')

# Định nghĩa hàm f(x) bằng sympy
f_sym = x_sym ** 3 - 3 * x_sym ** 2 + 6 * x_sym - 10

# Tính đạo hàm bằng sympy
f_derivative_sym = sp.diff(f_sym, x_sym)

print(f"\nHàm số: f(x) = {f_sym}")
print(f"Đạo hàm (tính bằng sympy): f'(x) = {f_derivative_sym}")

# Chuyển từ sympy sang hàm numpy để vẽ đồ thị
f = sp.lambdify(x_sym, f_sym, 'numpy')
f_derivative = sp.lambdify(x_sym, f_derivative_sym, 'numpy')

# Tính một số giá trị mẫu
print("\nMột số giá trị mẫu:")
sample_x = [-100, -50, 0, 50, 100]
for xi in sample_x:
    print(f"  x = {xi:4d}: f(x) = {f(xi):>12.2f}, f'(x) = {f_derivative(xi):>10.2f}")

# Vẽ đồ thị
x_vals = np.linspace(-100, 100, 1000)
y_vals = f(x_vals)
y_derivative_vals = f_derivative(x_vals)

# Tạo figure với 2 subplot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Đồ thị hàm f(x)
ax1.plot(x_vals, y_vals, 'b-', linewidth=2, label=r'$f(x) = x^3 - 3x^2 + 6x - 10$')
ax1.axhline(y=0, color='k', linewidth=0.5)
ax1.axvline(x=0, color='k', linewidth=0.5)
ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('f(x)', fontsize=12)
ax1.set_title(r'Đồ thị hàm $f(x) = x^3 - 3x^2 + 6x - 10$', fontsize=14)
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

# Đồ thị đạo hàm f'(x)
ax2.plot(x_vals, y_derivative_vals, 'r-', linewidth=2, label=r"$f'(x) = 3x^2 - 6x + 6$")
ax2.axhline(y=0, color='k', linewidth=0.5)
ax2.axvline(x=0, color='k', linewidth=0.5)
ax2.set_xlabel('x', fontsize=12)
ax2.set_ylabel("f'(x)", fontsize=12)
ax2.set_title(r"Đồ thị đạo hàm $f'(x) = 3x^2 - 6x + 6$", fontsize=14)
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/minh/Documents/python/CCTTKH/final_test/bai1_dothi.png', dpi=150)
plt.show()

print("\nĐồ thị đã được lưu tại: bai1_dothi.png")
print("\n" + "=" * 60)
print("HOÀN THÀNH BÀI 1")
print("=" * 60)

# ============================================================
# BÀI 2: XỬ LÝ DỮ LIỆU WINE.CSV
# ============================================================
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer, LabelEncoder
from scipy.fft import fft, fft2

df = pd.read_csv('../data/Wine.csv')

print("\n" + "=" * 60)
print("BÀI 2: XỬ LÝ DỮ LIỆU WINE.CSV")
print("=" * 60)

print("\nDữ liệu gốc (5 dòng đầu):")
print(df.head())
print(f"\nKích thước dữ liệu: {df.shape}")
print(f"Các cột: {df.columns.tolist()}")

# Lấy tên các cột
columns = df.columns.tolist()
col_1 = columns[0]  # Cột số 1
col_2 = columns[1]  # Cột số 2
col_3 = columns[2]  # Cột số 3
col_last = columns[-1]  # Cột cuối cùng

# a. Chuẩn hóa cột số 1 bằng MinMaxScaler [0, 1]
print("\n" + "-" * 60)
print("a. Chuẩn hóa cột 1 bằng MinMaxScaler [0, 1]")
print("-" * 60)

minmax_scaler = MinMaxScaler(feature_range=(0, 1))
df_col1_scaled = minmax_scaler.fit_transform(df[[col_1]])

print(f"\nCột '{col_1}' trước khi chuẩn hóa:")
print(f"  Min = {df[col_1].min()}, Max = {df[col_1].max()}")
print(f"\nCột '{col_1}' sau khi MinMaxScaler:")
print(f"  Min = {df_col1_scaled.min():.4f}, Max = {df_col1_scaled.max():.4f}")
print(f"\n5 giá trị đầu: {df_col1_scaled[:5].flatten()}")

# b. Chuẩn hóa cột 2 bằng StandardScaler (Standardize Data)
print("\n" + "-" * 60)
print("b. Chuẩn hóa cột 2 bằng StandardScaler (Standardize Data)")
print("-" * 60)

standard_scaler = StandardScaler()
df_col2_scaled = standard_scaler.fit_transform(df[[col_2]])

print(f"\nCột '{col_2}' trước khi chuẩn hóa:")
print(f"  Mean = {df[col_2].mean():.4f}, Std = {df[col_2].std():.4f}")
print(f"\nCột '{col_2}' sau khi StandardScaler:")
print(f"  Mean = {df_col2_scaled.mean():.6f}, Std = {df_col2_scaled.std():.4f}")
print(f"\n5 giá trị đầu: {df_col2_scaled[:5].flatten()}")

# c. Chuẩn hóa cột 3 bằng Normalize Data (L2 normalization)
print("\n" + "-" * 60)
print("c. Chuẩn hóa cột 3 bằng Normalize Data (L2 Normalization)")
print("-" * 60)

normalizer = Normalizer(norm='l2')
# Normalizer cần ít nhất 2D, reshape dữ liệu
df_col3_normalized = normalizer.fit_transform(df[[col_3]])

print(f"\nCột '{col_3}' trước khi chuẩn hóa:")
print(f"  5 giá trị đầu: {df[col_3].values[:5]}")
print(f"\nCột '{col_3}' sau khi Normalize (L2):")
print(f"  5 giá trị đầu: {df_col3_normalized[:5].flatten()}")

# d. Biến đổi Fourier Transform
print("\n" + "-" * 60)
print("d. Biến đổi Fourier Transform trên bộ dữ liệu")
print("-" * 60)

# Lấy các cột số (loại bỏ cột cuối nếu là class)
numeric_cols = columns[:-1] if df[columns[-1]].dtype == 'object' else columns
data_numeric = df[numeric_cols].values

# Áp dụng FFT 2D trên dữ liệu
print("\nÁp dụng FFT 2D (scipy.fft.fft2) trên dữ liệu số:")
fft_data = fft2(data_numeric)

print(f"\nKích thước dữ liệu gốc: {data_numeric.shape}")
print(f"Kích thước sau FFT: {fft_data.shape}")
print(f"\n5x5 giá trị đầu của FFT (phần thực):")
print(np.round(fft_data[:5, :5].real, 4))
print(f"\n5x5 giá trị đầu của FFT (phần ảo):")
print(np.round(fft_data[:5, :5].imag, 4))

# Áp dụng FFT 1D trên cột đầu tiên
print(f"\nÁp dụng FFT 1D trên cột '{col_1}':")
fft_col1 = fft(df[col_1].values)
print(f"5 giá trị đầu: {np.round(fft_col1[:5], 4)}")

# e. Số hóa cột cuối cùng (Label Encoding)
print("\n" + "-" * 60)
print("e. Số hóa cột cuối cùng bằng Label Encoding")
print("-" * 60)

label_encoder = LabelEncoder()
df_last_encoded = label_encoder.fit_transform(df[col_last])

print(f"\nCột '{col_last}' trước khi số hóa:")
print(f"  Giá trị unique: {df[col_last].unique()}")
print(f"  5 giá trị đầu: {df[col_last].values[:5]}")

print(f"\nCột '{col_last}' sau khi Label Encoding:")
print(f"  Classes: {label_encoder.classes_}")
print(f"  5 giá trị đầu: {df_last_encoded[:5]}")

# Tổng hợp kết quả
print("\n" + "-" * 60)
print("TỔNG HỢP KẾT QUẢ BÀI 2")
print("-" * 60)

# Tạo DataFrame kết quả
df_result = pd.DataFrame({
    f'{col_1}_MinMax': df_col1_scaled.flatten(),
    f'{col_2}_Standard': df_col2_scaled.flatten(),
    f'{col_3}_Normalize': df_col3_normalized.flatten(),
    f'{col_last}_Encoded': df_last_encoded
})

print("\nDữ liệu sau khi xử lý (5 dòng đầu):")
print(df_result.head())

print("\n" + "=" * 60)
print("HOÀN THÀNH BÀI 2")
print("=" * 60)
