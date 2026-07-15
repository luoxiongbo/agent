import numpy as np

scores = np.array([55, 68, 72, 90, 45, 88])

print(scores >= 60)
print(scores >= 70)

print(scores[scores >= 60])
print(scores)

print(scores[scores < 60])
scores[scores < 60] = 0
print(scores)

# 注意：NumPy 里多个条件不能直接写 and，要用 &，并且每个条件要加括号。
print(scores[(scores >= 60) & (scores < 80)])

a = np.array([
    [55, 68, 72],
    [90, 45, 88]
])

# 注意：布尔筛选二维数组时，筛出来通常会变成一维结果。
print(a[a < 60])
