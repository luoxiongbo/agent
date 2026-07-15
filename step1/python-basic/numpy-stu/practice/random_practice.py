import numpy as np

rng = np.random.default_rng(42)

rng.integers(1, 10, size=5)
rng.integers(60, 101, size=(4, 3))
rng.random(5)
rng.random((2, 3))
rng.normal(70, 10, size=10)
x = np.array(["A", "B", "C", "D", "E"])
rng.choice(x, size=3)
x = np.array([1, 2, 3, 4, 5])
rng.shuffle(x)
print(x)

scores = rng.integers(0, 101, size=(5, 4))
print(scores)
print(scores.mean(axis=1))
print(scores.mean(axis=0))
print(scores[scores < 60])