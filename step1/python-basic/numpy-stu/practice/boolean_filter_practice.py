import numpy as np

scores = np.array([55, 68, 72, 90, 45, 88, 100, 77])

print(scores >= 60)
print(scores[scores >= 60])
print(scores[scores < 60])
print(scores[scores >= 80])
print(scores[(60 <= scores) & (scores < 80)])

m = np.array([
    [55, 68, 72],
    [90, 45, 88],
    [100, 77, 59]
])
print(m < 60)
print(m[m < 60])
print(m[m >= 80])
m[m < 60] = 0
print(m)
