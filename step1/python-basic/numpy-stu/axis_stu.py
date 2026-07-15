import numpy as np

a = np.array([
    [10, 20, 30],
    [40, 50, 60]
])

print(a.shape)
print(a.sum())
print(a.sum(axis=0))
print(a.sum(axis=1))

print(a.mean(axis=0))
print(a.mean(axis=1))

print(a.max(axis=0))
print(a.max(axis=1))

print(a.min(axis=0))
print(a.min(axis=1))

# axis=0 是按列计算