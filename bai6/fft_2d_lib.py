import numpy as np

S = np.array([
    [8, 1, 8, 1],
    [1, 8, 1, 8],
    [8, 1, 8, 1],
    [1, 8, 1, 8]
])

S_dft = np.fft.fft2(S, norm='ortho')

print("Ma trận S ban đầu: ")
print(S)
print("\n" + "=" * 30)
print("Kết quả sau khi biến đổi FFT: ")
print(np.round(S_dft, decimals=4))

import numpy as np

S_dft = np.array([
    [18. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
    [0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
    [0. + 0.j, 0. + 0.j, 14. + 0.j, 0. + 0.j],
    [0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j]
])

S_restored = np.fft.ifft2(S_dft, norm='ortho')

S_final = np.round(S_restored.real).astype(int)

print("Ma trận S_dft (đầu vào):")
print(np.round(S_dft, 2))
print("\n" + "=" * 30)
print("Kết quả sau khi biến đổi ngược (ma trận S khôi phục):")
print(S_final)
