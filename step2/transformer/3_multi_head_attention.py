import math

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
手写多头注意力
Transformer 中最重要的代码。
"""

class MultiHeadSelfAttention(nn.Module):

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError("d_model 必须能被 num_heads 整除")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # 一次性计算 Q、K、V，效率比创建三个 Linear 更高。
        self.qkv = nn.Linear(
            d_model,
            3 * d_model,
            bias=False,
        )

        # 多个头拼接后，再进行一次线性变换。
        self.output_projection = nn.Linear(
            d_model,
            d_model,
            bias=False,
        )

    def forward(self, x):
        """
        x.shape:
            [batch_size, sequence_length, d_model]
        """

        batch_size, sequence_length, d_model = x.shape

        # qkv.shape:
        # [batch, sequence, 3 * d_model]
        qkv = self.qkv(x)

        # 拆成 Q、K、V。
        #
        # 每一个的 shape 都是：
        # [batch, sequence, d_model]
        q, k, v = qkv.chunk(3, dim=-1)

        # 将 d_model 拆成：
        # num_heads × head_dim
        #
        # 变化前：
        # [batch, sequence, d_model]
        #
        # 变化后：
        # [batch, sequence, heads, head_dim]
        q = q.reshape(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        )

        k = k.reshape(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        )

        v = v.reshape(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        )

        # 调整维度顺序：
        #
        # [batch, sequence, heads, head_dim]
        #                ↓
        # [batch, heads, sequence, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # k.transpose(-2, -1):
        #
        # [batch, heads, sequence, head_dim]
        #                ↓
        # [batch, heads, head_dim, sequence]
        #
        # attention_scores.shape:
        # [batch, heads, sequence, sequence]
        attention_scores = (
            q @ k.transpose(-2, -1)
        ) / math.sqrt(self.head_dim)

        attention_weights = F.softmax(
            attention_scores,
            dim=-1,
        )

        # 每个头分别对 V 加权求和。
        #
        # output.shape:
        # [batch, heads, sequence, head_dim]
        output = attention_weights @ v

        # 将 heads 放回后面。
        #
        # [batch, heads, sequence, head_dim]
        #                ↓
        # [batch, sequence, heads, head_dim]
        output = output.transpose(1, 2)

        # 将多个头重新拼接成 d_model。
        #
        # [batch, sequence, heads, head_dim]
        #                ↓
        # [batch, sequence, d_model]
        output = output.contiguous().reshape(
            batch_size,
            sequence_length,
            d_model,
        )

        output = self.output_projection(output)

        return output, attention_weights


if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size = 2
    sequence_length = 5
    d_model = 8
    num_heads = 2

    x = torch.randn(
        batch_size,
        sequence_length,
        d_model,
    )

    attention = MultiHeadSelfAttention(
        d_model=d_model,
        num_heads=num_heads,
    )

    output, weights = attention(x)

    print("输入 shape：")
    print(x.shape)

    print("\n输出 shape：")
    print(output.shape)

    print("\n注意力权重 shape：")
    print(weights.shape)

    print("\n第一个样本、第一个头的注意力矩阵：")
    print(weights[0, 0])

    print("\n每行权重之和：")
    print(weights[0, 0].sum(dim=-1))

    """
    为什么输出维度没有变？
    只是在 token 原本的信息上 + 其他 token 的信息
    给 token 自身加上它在上下文中的语义
    """