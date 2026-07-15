import numpy as np

rng = np.random.default_rng(42)

# 6 个学生，4 门课
scores = rng.integers(0, 101, size=(6, 4))

print(scores)

# 1. 输出 scores 的 shape
print(scores.shape)
# 2. 输出每个学生的平均分
print(scores.mean(axis=1))
# 3. 输出每门课的平均分
print(scores.mean(axis=0))
# 4. 输出每个学生的总分
print(scores.sum(axis=1))
# 5. 找出总分最高的学生编号
max = 0
result = 0
for i, score in enumerate(scores.sum(axis=1)):
    if score > max:
        max = score
        result = i
print(result)
# 6. 找出每门课最高分
print(scores.max(axis=0))
# 7. 找出所有不及格成绩，也就是 < 60 的分数
print(scores[scores < 60])
# 8. 统计每个学生有几门课不及格

# 9. 给所有成绩加 5 分，但最高不能超过 100
# 提示：np.clip(..., 0, 100)
print(np.clip(scores + 5, 0, 100))
# 10. 按平均分从高到低排序，输出学生编号
# 提示：np.argsort(...)
print(np.argsort(scores))