import math

import numpy as np
import pandas as pd

df = pd.read_csv('./data2.csv', header=None)
result = np.zeros((8, 8))

for j in range(0, 8):
    for k in range(0, 8):
        for u in range(0, 8):
            for v in range(0, 8):
                cu = (1 / math.sqrt(2)) if u == 0 else 1
                cv = (1 / math.sqrt(2)) if v == 0 else 1
                result[j, k] += df.at[u, v] * math.cos(((2 * j + 1) * u * math.pi) / 16) * math.cos(
                    ((2 * k + 1) * v * math.pi) / 16) * ((cu * cv) / 4)

print('Bien doi nghich: ')
print(np.round(result))
