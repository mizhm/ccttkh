import math

import numpy as np
import pandas as pd

df = pd.read_csv('./data.csv', header=None)
result = np.zeros((8, 8))

for u in range(0, 8):
    for v in range(0, 8):
        cu = (1 / math.sqrt(2)) if u == 0 else 1
        cv = (1 / math.sqrt(2)) if v == 0 else 1
        for j in range(0, 8):
            for k in range(0, 8):
                result[u, v] += df.at[j, k] * math.cos(((2 * j + 1) * u * math.pi) / 16) * math.cos(
                    ((2 * k + 1) * v * math.pi) / 16)
        result[u, v] *= (cu * cv) / 4

print('Bien doi thuan: ')
print(np.round(result))
