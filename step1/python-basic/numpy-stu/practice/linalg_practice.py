import numpy as np

A = np.array([
    [1, 2],
    [3, 4]
])

B = np.array([
    [5, 6],
    [7, 8]
])

print(A @ B)
print(B @ A)
print(A.T)
"""
行列式：判断矩阵是否可逆的重要指标。
逆矩阵：能把矩阵变换“撤销”的矩阵。
det(A) != 0，A 才有逆矩阵。
det(A) == 0，A 没有逆矩阵。
"""
print(np.linalg.det(A))
print(np.linalg.inv(A))
I = A @ np.linalg.inv(A)
print(I)
print(A @ I)
print(np.allclose(A @ np.linalg.inv(A), np.eye(2)))

C = np.array([
    [2, 1],
    [1, 3]
])

d = np.array([
    [8],
    [13]
])

print(np.linalg.solve(C, d))