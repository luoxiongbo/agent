import math

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
加入真正可学习的 Q、K、V
"""

torch.manual_seed(42)


X = torch.randn(3, 4)

# 4 维向量？
d_model = 4
#
d_k = 4


# 三个线性层拥有不同的可训练参数。
q_layer = nn.Linear(d_model, d_k, bias=False)
k_layer = nn.Linear(d_model, d_k, bias=False)
v_layer = nn.Linear(d_model, d_k, bias=False)

"""
Q = X @ Wq
K = X @ Wk
V = X @ Wv
"""
Q = q_layer(X)
K = k_layer(X)
V = v_layer(X)

"""
训练的时候，反向传播会自动更新这些参数
q_layer.weight
k_layer.weight
v_layer.weight
"""
scores = Q @ K.T / math.sqrt(d_k)
attention_weights = F.softmax(scores, dim=-1)
output = attention_weights @ V


print("X.shape:", X.shape)
print("Q.shape:", Q.shape)
print("K.shape:", K.shape)
print("V.shape:", V.shape)

print("\nattention_weights.shape:")
print(attention_weights.shape)

print("\n注意力权重：")
print(attention_weights)

print("\n输出：")
print(output)

print("\nq_layer的参数权重：")
print(q_layer.weight)
print("\nk_layer的参数权重：")
print(k_layer.weight)
print("\nv_layer的参数权重：")
print(v_layer.weight)