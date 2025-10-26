n = int(input('n = '))
while n <= 0:
    print('n phai la so nguyen duong')
    n = int(input('n = '))
B = 0
for k in range(1, n + 1):
    B += 1 / (k ** 2)
print('B = ', B)
