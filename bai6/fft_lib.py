import numpy as np
from scipy.fft import fft, fftshift

# Tín hiệu đầu vào
s1 = np.array([8, 1, 8, 1])
s2 = np.array([1, 8, 1, 8])

# Tính FFT tiêu chuẩn (tương ứng k = 0, 1, 2, 3)
S1_std_fft = fft(s1)
S2_std_fft = fft(s2)

# Sắp xếp lại kết quả để tương ứng với k = -2, -1, 0, 1
S1_shifted = fftshift(S1_std_fft)
S2_shifted = fftshift(S2_std_fft)

np.set_printoptions(suppress=True, precision=2)

print("--- KẾT QUẢ TỪ HÀM FFT CỦA SCIPY ---")
print("FFT của s1 (sau khi shift): \n", S1_shifted.reshape(-1, 1))
print("\nFFT của s2 (sau khi shift): \n", S2_shifted.reshape(-1, 1))
