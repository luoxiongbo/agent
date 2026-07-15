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
"""
[19, 22]
[43, 50]
"""

"""
解方程组
2x + y = 8
x + 3y = 13
"""
C = np.array([
    [2, 1],
    [1, 3]
])

# d = np.array([8, 13])
d = np.array([
    [8],
    [13]
])

u = np.linalg.solve(C, d)

print(u)
print(C @ u)
print(d)