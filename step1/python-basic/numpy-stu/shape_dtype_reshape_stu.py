import numpy as np

a = np.array([1, 2, 3, 4, 5, 6])
b = np.array([
    [1],
    [2],
    [3],
    [4],
    [5],
    [6]
])
c = np.dot(a, b)
print(a.shape)
print(b.shape)
print(c)

d = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9.0],
])
print(d.dtype)

e = np.reshape(d, (1, 9))
print(e)


