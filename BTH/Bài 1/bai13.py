import math

x = float(input('x = '))
n = float(input('n = '))
print('D = ', round((math.sin(n * x)) / (1 + math.cos(x) ** 2) + (x + 1) ** (1 / n), 2))
