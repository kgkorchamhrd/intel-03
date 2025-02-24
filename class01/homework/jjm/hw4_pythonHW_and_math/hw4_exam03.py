# exam03
# 실습 2에서 수행한 결과를 reshape를 이용해서 1차원 형태로 변환한다.

import numpy as np

n = int(input())

A = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        A[i, j] = n * i + j

print(A)

B = A.reshape(-1,)



# n = 4

# list01 = []
# list02 = []

# for i in range(n * n + 1):      # + 1 => need append
#     j = i + 1
#     if i > 0 and i % n == 0:
#         list01.append(list02)
#         list02 = []
#     list02.append(j)

# list03 = np.reshape(list01, (1, n * n))
# print(list03)
