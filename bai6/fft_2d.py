import cmath
import numpy as np

S = [
    [8, 1, 8, 1],
    [1, 8, 1, 8],
    [8, 1, 8, 1],
    [1, 8, 1, 8]
]

M = len(S)
N = len(S[0])

S_dft = np.zeros((M, N), dtype=complex)

for u in range(M):
    for v in range(N):

        current_sum = 0

        for m in range(M):
            for n in range(N):
                exponent = -2j * cmath.pi * ((u * m / M) + (v * n / N))
                current_sum += S[m][n] * cmath.exp(exponent)

        S_dft[u, v] = (1 / cmath.sqrt(M * N)) * current_sum

print("Kết quả sau khi biến đổi FFT:")
print(np.round(S_dft, decimals=4))
