import numpy as np

i = int(input("enter a digit:"))

arr = np.zeros((i,i))
cur_num = 1

for a in range(i):
    for b in range(i):
        arr[a][b] = cur_num
        cur_num += 1
print(arr)

reshaped_arr = arr.reshape(-1)
print(reshaped_arr)