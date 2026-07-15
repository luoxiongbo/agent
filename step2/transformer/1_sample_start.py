import math

import torch
import torch.nn.functional as F


"""
第一份可执行代码：手写单头注意力
为了理解注意力公式
"""

torch.set_printoptions(precision=3, sci_mode=False)

# 假设一句话中有三个 token。
# 每个 token 使用一个 4 维向量表示。
#
# X.shape = [序列长度, 向量维度]
X = torch.tensor([
    [1.0, 0.0, 1.0, 0.0],  # token 0
    [1.0, 0.0, 0.0, 0.0],  # token 1
    [0.0, 1.0, 0.0, 1.0],  # token 2
])

# 为了先理解注意力公式，
# 暂时假设 Wq、Wk、Wv 都是单位矩阵。
#
# 所以 Q = K = V = X。
#
# 真正的 Transformer 中，
# Q、K、V 会分别经过三个可训练的线性层。
Q = X
K = X
V = X

# 每个 Query 和所有 Key 计算点积。
#
# Q.shape   = [3, 4]
# K.T.shape = [4, 3]
# scores.shape = [3, 3]
scores = Q @ K.T

# 缩放，避免点积数值随着维度增大而过大。
d_k = K.size(-1)
scores = scores / math.sqrt(d_k)


# 对每一行做 softmax。
#
# dim=-1 表示对每个 token 关注所有 token 的分数 进行归一化。
# 结果的每一行表示一个 token 对所有 token 的关注程度。
attention_weights = F.softmax(scores, dim=-1)


# 使用注意力权重，对 V 进行加权求和。
"""
shape 变化：[3, 3] * [3, 4] = [3, 4]
注意力点乘向量，也就是把注意力信息组合到向量里
output 每行是第 i 个 token 汇总上下文信息后得到的新的四维向量
"""
output = attention_weights @ V


print("输入 X：")
print(X)

print("\n注意力原始分数：")
print(scores)

print("\n注意力权重：")
print(attention_weights)

print("\n注意力输出：")
print(output)

print("\n每一行注意力权重之和：")
print(attention_weights.sum(dim=-1))