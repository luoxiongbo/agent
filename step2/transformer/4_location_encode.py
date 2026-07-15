import torch
import torch.nn as nn

"""
位置编码
"""

vocab_size = 10000
max_sequence_length = 128
d_model = 512


token_embedding = nn.Embedding(
    vocab_size,
    d_model,
)

position_embedding = nn.Embedding(
    max_sequence_length,
    d_model,
)


token_ids = torch.tensor([
    [10, 25, 98, 36],
    [17, 42, 11, 90],
])

batch_size, sequence_length = token_ids.shape


positions = torch.arange(sequence_length)

print("positions：")
print(positions)


token_vectors = token_embedding(token_ids)
position_vectors = position_embedding(positions)


x = token_vectors + position_vectors


print("token_vectors.shape：")
print(token_vectors.shape)

print("position_vectors.shape：")
print(position_vectors.shape)

print("最终输入 shape：")
print(x.shape)