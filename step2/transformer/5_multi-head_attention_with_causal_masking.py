import math

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
带因果掩码的多头注意力
"""


class CausalSelfAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError("d_model 必须能被 num_heads 整除")

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.qkv = nn.Linear(
            d_model,
            3 * d_model,
            bias=False,
        )

        self.output_projection = nn.Linear(
            d_model,
            d_model,
            bias=False,
        )

    def forward(self, x, return_attention=False):
        batch_size, sequence_length, d_model = x.shape

        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def split_heads(tensor):
            return tensor.reshape(
                batch_size,
                sequence_length,
                self.num_heads,
                self.head_dim,
            ).transpose(1, 2)

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)

        scores = (
            q @ k.transpose(-2, -1)
        ) / math.sqrt(self.head_dim)

        # 上三角矩阵，不包含对角线。
        #
        # True 表示这些位置是未来信息，需要屏蔽。
        causal_mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                device=x.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )

        scores = scores.masked_fill(
            causal_mask,
            float("-inf"),
        )

        attention_weights = F.softmax(
            scores,
            dim=-1,
        )

        output = attention_weights @ v

        output = output.transpose(
            1,
            2,
        ).contiguous().reshape(
            batch_size,
            sequence_length,
            d_model,
        )

        output = self.output_projection(output)

        if return_attention:
            return output, attention_weights

        return output


if __name__ == "__main__":
    torch.manual_seed(42)

    x = torch.randn(1, 4, 8)

    attention = CausalSelfAttention(
        d_model=8,
        num_heads=2,
    )

    output, weights = attention(
        x,
        return_attention=True,
    )

    print("第一个头的注意力矩阵：")
    print(weights[0, 0])