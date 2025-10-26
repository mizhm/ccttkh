import numpy as np

N = 4
indices = np.array([-2, -1, 0, 1])

k = indices.reshape(-1, 1)
n = indices

kn_matrix = k * n

W_matrix = np.exp(-1j * 2 * np.pi * kn_matrix / N)

np.set_printoptions(suppress=True, precision=2)

print("Ma trận hệ số DFT (W):")
print(W_matrix)

s1 = np.array([8, 1, 8, 1]).reshape(-1, 1)
s2 = np.array([1, 8, 1, 8]).reshape(-1, 1)

S1_fft = 1 / np.sqrt(N) * (W_matrix @ s1)
S2_fft = 1 / np.sqrt(N) * (W_matrix @ s2)

print("\nFFT của s1:\n", S1_fft)
print("\nFFT của s2:\n", S2_fft)

W_inv_matrix = W_matrix.conj()
print("\nMa trận biến đổi ngược (W_conj):")
print(W_inv_matrix)

s1_recovered = (1 / np.sqrt(N)) * (W_inv_matrix @ S1_fft)
s2_recovered = (1 / np.sqrt(N)) * (W_inv_matrix @ S2_fft)

print("\n--- KẾT QUẢ KHÔI PHỤC ---")
print("Tín hiệu s1 gốc: \n", s1)
print("\nTín hiệu s1 khôi phục từ IDFT: \n", s1_recovered)
print("\n------------------------------")
print("Tín hiệu s2 gốc: \n", s2)
print("\nTín hiệu s2 khôi phục từ IDFT: \n", s2_recovered)
