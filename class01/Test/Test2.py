import numpy as np

n = int(input())
A = np.zeros((n,n))

for i in range(n):
    for j in range(n):
        A[i,j] = n*1 + j
print(A)