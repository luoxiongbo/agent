import numpy as np

rng = np.random.default_rng()
print(rng.integers(1, 10, size=5))

print(rng.integers(60, 101, size=(4, 3)))

print(rng.random(5))
print(rng.random((2, 3)))

"""
正态分布
loc：均值
scale：标准差
size：形状
"""
print(rng.normal(loc=0, scale=1, size=5))

# 生成一批接近 70 分、波动 10 分的成绩：
scores = rng.normal(loc=70, scale=10, size=10)
print(scores)

rng = np.random.default_rng(42)
print("每次结果都一样: ", rng.integers(1, 10, size=5))


names = np.array(["A", "B", "C", "D"])
print(rng.choice(names, size=2))

x = np.array([1, 2, 3, 4, 5])
rng.shuffle(x)
print(x)