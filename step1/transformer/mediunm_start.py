import math

import torch
import torch.nn as nn
import torch.nn.functional as F


torch.manual_seed(42)


X = torch.randn(3, 4)

d_model = 4
d_k = 4


# 三个线性层拥有不同的可训练参数。
q_layer = nn.Linear(d_model, d_k, bias=False)
k_layer = nn.Linear(d_model, d_k, bias=False)
v_layer = nn.Linear(d_model, d_k, bias=False)


Q = q_layer(X)
K = k_layer(X)
V = v_layer(X)


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