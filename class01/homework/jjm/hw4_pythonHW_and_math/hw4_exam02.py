# exam02
# 정수 n을 입력받아 n x n 크기의 숫자 직사각형 출력.
import numpy as np

n = int(input())

A = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        A[i, j] = n * i + j

print(A)


# n = 4

# for i in range(n * n):
#     j = i + 1
#     if i > 0 and i % n == 0:
#         print(" ")
#     print(j, end = " ")
