import numpy as np

# 4 个学生，3 门课
scores = np.array([
    [80, 90, 100],
    [70, 60, 75],
    [88, 92, 85],
    [50, 55, 40]
])

print(scores.sum())
print(scores.sum(axis=0))
print(scores.sum(axis=1))
print(scores.mean())
print(scores.mean(axis=0))
print(scores.mean(axis=1))
print(scores.max(axis=0))
print(scores.max(axis=1))
print(scores.min(axis=0))
print(scores.min(axis=1))