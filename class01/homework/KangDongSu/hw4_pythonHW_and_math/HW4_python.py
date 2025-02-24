import numpy as np  # NumPy 라이브러리 가져오기

# 정수 n 입력받기
n = int(input("n을 입력하세요: "))

# n x n 숫자 행렬 생성
matrix = np.arange(1, n*n + 1).reshape(n, n)  # 2차원 배열 생성

# 결과 출력
print("2차원 배열:")
print(matrix)

# 1차원 배열로 변환
flattened = matrix.reshape(-1)

# 결과 출력
print("\n1차원 배열:")
print(flattened)
