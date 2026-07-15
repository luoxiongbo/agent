import numpy as np

a = np.array([1, 2, 3, 4, 5])
b = np.array([6, 7, 8, 9, 10])

print(a)
print(b)
print(a.mean())
print(a.max())
print(a.min())
print(a.sum())

print(a + b)
print(b - a)
print(a * b)

c = np.array([
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10]
])

d = np.array([
    [1, 2],
    [3, 4],
    [5, 6],
    [7, 8],
    [9, 10]
])
print(c.shape)
print(d.shape)
result = c @ d
print(result[0])
print(result[1])
"""
[ 95, 110 ]
[ 220, 260 ]
"""

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
print(x)
print(x.shape)
print(x.dtype)
a = np.reshape(x, (3, 4))
print(a)
print(a.shape)
b = np.reshape(x, (4, 3))
print(b)
print(b.shape)
c = np.reshape(x, (2, -1))

d = np.array([1, 2, 3.5])
print(d.dtype)
e = np.array([1, 2, 3], dtype=float)
print(e)
print(e.dtype)

