import numpy as np

i = int(input("enter a digit:"))

arr = np.zeros((i,i))

for a in range(i):
    for b in range(i):
        arr[a][b] = i*a + b + 1
print(arr)
