import numpy as np

# 2번
num = input("정수입력")

temp = 0
for i in range(0, int(num)):
    for j in range(0, int(num)):
        temp = temp + 1
        print(f"{temp}", end=" ")
    print()

# 2번 다른거
n = int(input())
A = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        A[i, j] = n*i + j


# 3번
B = A.reshape(-1, )
