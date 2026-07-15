# 广播的意思是：当两个数组形状不完全一样时，只要规则允许，NumPy 会自动把小数组“扩展”成合适的形状再计算。
import numpy as np

import numpy as np

x = np.array([10, 20, 30])
print(x + 5)
# 这里会把 5 当作 [5, 5, 5]

m = np.array([
    [10, 20, 30],
    [40, 50, 60]
])

v = np.array([1, 2, 3])

print(m + v)

col = np.array([
    [100],
    [200]
])

print(m + col)