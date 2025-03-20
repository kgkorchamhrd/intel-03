import numpy as np

A = np.array([[1, 4, 2, 0], [9, 5, 0, 0], [4, 0, 2, 4], [6, 1, 8, 3]])
b = np.array([15, 19, 26, 44])

print("det=", np.linalg.det(A))

x = np.linalg.solve(A, b)
print("solver = ", x)

x = np.dot(np.linalg.inv(A), b)
print("inverse1 = ", x)

tmp_b = np.dot(A.T, b)
tmp_T = np.dot(A.T, A)
tmp_inv = np.linalg.inv(np.dot(A.T, A))
x = np.dot(tmp_inv, tmp_b)
print("inverse2 = ", x)
