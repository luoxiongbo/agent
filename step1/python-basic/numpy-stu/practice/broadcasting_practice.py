import numpy as np

m = np.array([
    [10, 20, 30],
    [40, 50, 60]
])

print(m + 5)

v = np.array([1, 2, 3])
print(m + v)

col = np.array([
    [100],
    [200],
])
print(m + col)

print(m * v)
print(m * col)

print(m.shape)
print(v.shape)
print(col.shape)

bad = np.array([1, 2])
print(m + bad)