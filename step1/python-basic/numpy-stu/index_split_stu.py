import numpy as np

a = np.array([
    [10, 20, 30],
    [40, 50, 60],
    [70, 80, 90]
])

print(a.shape)
print(a[1, 1])
# 冒号 : 表示“这一维全部”
print(a[0, :])
print(a[0])
print(a[:, 0])

print(a[0:2, 1:3])

