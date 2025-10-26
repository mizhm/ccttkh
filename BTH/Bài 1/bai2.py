m = int(input("m: "))
while m <= 0:
    print("m phai la so nguyen duong")
    m = int(input("m: "))
n = int(input("n: "))
while n > m or n <= 0:
    print("n phai la so nguyen duong va <= m")
    n = int(input("n: "))

print('Phan nguyen khi chia m cho n: ', m // n)
print('Phan du khi chia m cho n: ', m % n)
