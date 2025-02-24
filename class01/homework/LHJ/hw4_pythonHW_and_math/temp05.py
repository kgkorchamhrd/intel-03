import numpy as np

from numpy import linalg as LA

c = np.array([[1, 2, 3], [-1, 1, 4]])

print(LA.norm(c, axis=0))
print(LA.norm(c, axis=1))
print(LA.norm(c, ord=1, axis=1))  # 열끼리 더함
print(LA.norm(c, ord=2, axis=1))  # 열끼리 제곱해서 더한다음 루트
