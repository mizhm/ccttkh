import numpy as np
from scipy.fftpack import dct, idct

# 1. Tạo một khối ảnh 4x4 giả lập (ví dụ: một mẫu sọc đơn giản)
# Các giá trị pixel từ 0-255
block = np.array([
    [200, 200, 50, 50],
    [200, 200, 50, 50],
    [200, 200, 50, 50],
    [200, 200, 50, 50]
], dtype=float)

print("--- Khối ảnh gốc ---")
print(block)


# 2. Hàm thực hiện 2D DCT
def dct2(a):
    # axis=0: làm trên hàng, axis=1: làm trên cột (tính tách biệt)
    # norm='ortho': chuẩn hóa để ma trận trực giao
    return dct(dct(a.T, norm='ortho').T, norm='ortho')


# 3. Hàm thực hiện 2D IDCT
def idct2(a):
    return idct(idct(a.T, norm='ortho').T, norm='ortho')


# --- THỰC HIỆN ---
dct_block = dct2(block)
print("\n--- Ma trận DCT (Miền tần số) ---")
np.set_printoptions(precision=1, suppress=True)
print(dct_block)

reconstructed_block = idct2(dct_block)
print("\n--- Ảnh tái tạo sau IDCT ---")
print(reconstructed_block)
