import numpy as np

A = np.array([[1, 4, 2, 0], [9, 5, 0, 0], [4, 0, 2, 4], [6, 1, 8, 3]])
B = np.array([15, 19, 26, 44])

print("det = ", np.linalg.det(A))

x =  np.linalg.solve(A, B)
print("solver = ", x)

x = np.dot(np.linalg.inv(A), B)
print("inverse_01 = ", x)

temp_B = np.dot(A.T, B)
temp_T = np.dot(A.T, A)
temp_inv = np.linalg.inv(np.dot(A.T, A))
x = np.dot(temp_inv, temp_B)
print("inverse_02 = ", x)




# import numpy as np

# A = np.array([[1, 4, 2, 0], [9, 5, 0, 0], [4, 0, 2, 4], [6, 1, 8, 3]])
# x = np.array([1, 2, 3, 4])
# B = np.array([0, 0, 0, 0])
# n = 4

# for i in range(0, n):
#     val = 0.0
#     for j in range(0, n):
#         val += A[i, j] * x[j]
#     B[i] = val

# print("catculate = ", B)

# B = np.dot(A, x)
# print("dot = ", B)

# B= np.matmul(A, x)
# print("matmul = ", B)

# B= A @ x
# print("A @ x = ", B)

# B = A * x
# print("A * x = ", B)



# import numpy as np

# a = np.array([[1], [2], [3], [4]])

# print(a.T)




# import numpy as np
# from numpy import linalg as LA

# c = np.array([[1, 2, 3],
#               [-1, 1, 4]])
# print(LA.norm(c, axis = 0))             # 노말리제이션
# print(LA.norm(c, axis = 1))
# print(LA.norm(c, ord = 1, axis = 1))    # 오더 1, 노말리제이션
# print(LA.norm(c, ord = 2, axis = 1))




