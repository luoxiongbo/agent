import math

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
完整可训练的小型 Transformer
"""

# 避免某些电脑上小模型多线程反而运行缓慢。
torch.set_num_threads(1)
torch.manual_seed(42)

# 多头注意力
class CausalSelfAttention(nn.Module):

    def __init__(
        self,
        d_model=16,
        num_heads=4,
    ):
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

    def forward(self, x):
        batch_size, sequence_length, d_model = x.shape

        # 一次得到 Q、K、V。
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        # [B, T, C]
        #     ↓
        # [B, H, T, D]
        q = q.reshape(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        k = k.reshape(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        v = v.reshape(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        # 注意力分数：
        # [B, H, T, T]
        scores = (
            q @ k.transpose(-2, -1)
        ) / math.sqrt(self.head_dim)

        # 屏蔽未来位置。
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

        # 根据注意力权重汇总 V。
        output = attention_weights @ v

        # [B, H, T, D]
        #     ↓
        # [B, T, C]
        output = output.transpose(
            1,
            2,
        ).contiguous().reshape(
            batch_size,
            sequence_length,
            d_model,
        )

        return self.output_projection(output)


class FeedForward(nn.Module):

    def __init__(self, d_model=16):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model),
        )

    def forward(self, x):
        return self.network(x)


class TransformerBlock(nn.Module):

    def __init__(
        self,
        d_model=16,
        num_heads=4,
    ):
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.attention = CausalSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
        )

        self.layer_norm_2 = nn.LayerNorm(d_model)
        self.feed_forward = FeedForward(
            d_model=d_model,
        )

    def forward(self, x):
        # 残差连接 + 注意力。
        x = x + self.attention(
            self.layer_norm_1(x)
        )

        # 残差连接 + 前馈网络。
        x = x + self.feed_forward(
            self.layer_norm_2(x)
        )

        return x


class TinyTransformer(nn.Module):

    def __init__(
        self,
        vocab_size=10,
        context_length=8,
        d_model=16,
        num_heads=4,
    ):
        super().__init__()

        self.context_length = context_length

        # 数字 token 的嵌入。
        self.token_embedding = nn.Embedding(
            vocab_size,
            d_model,
        )

        # 位置嵌入。
        self.position_embedding = nn.Embedding(
            context_length,
            d_model,
        )

        # 一个 Transformer Block。
        self.block = TransformerBlock(
            d_model=d_model,
            num_heads=num_heads,
        )

        self.final_layer_norm = nn.LayerNorm(
            d_model
        )

        # 把每个位置的向量转换成 10 个数字的预测分数。
        self.output_head = nn.Linear(
            d_model,
            vocab_size,
        )

    def forward(self, token_ids, targets=None):
        """
        token_ids.shape:
            [batch_size, sequence_length]
        """

        _, sequence_length = token_ids.shape

        positions = torch.arange(
            sequence_length,
            device=token_ids.device,
        )

        # 词嵌入 + 位置嵌入。
        x = (
            self.token_embedding(token_ids)
            + self.position_embedding(positions)
        )

        x = self.block(x)
        x = self.final_layer_norm(x)

        # logits.shape:
        # [batch_size, sequence_length, vocab_size]
        logits = self.output_head(x)

        loss = None

        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(
                    -1,
                    logits.size(-1),
                ),
                targets.reshape(-1),
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        token_ids,
        max_new_tokens,
    ):
        for _ in range(max_new_tokens):

            # 最多只保留 context_length 个 token。
            current_tokens = token_ids[
                :,
                -self.context_length:
            ]

            logits, _ = self(current_tokens)

            # 只取最后一个位置的预测。
            next_token_logits = logits[:, -1]

            # 选择概率最大的 token。
            next_token = next_token_logits.argmax(
                dim=-1,
                keepdim=True,
            )

            token_ids = torch.cat(
                [token_ids, next_token],
                dim=1,
            )

        return token_ids


def make_batch(
    batch_size=16,
    sequence_length=8,
):
    """
    随机生成递增数字序列。

    例如：
        输入  3 4 5 6 7 8 9 0
        标签  4 5 6 7 8 9 0 1
    """

    start = torch.randint(
        low=0,
        high=10,
        size=(batch_size, 1),
    )

    steps = torch.arange(
        sequence_length + 1
    ).unsqueeze(0)

    sequence = (start + steps) % 10

    inputs = sequence[:, :-1]
    targets = sequence[:, 1:]

    return inputs, targets


model = TinyTransformer(
    vocab_size=10,
    context_length=8,
    d_model=16,
    num_heads=4,
)


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.01,
)


# 开始训练。
for step in range(101):
    inputs, targets = make_batch()

    _, loss = model(
        inputs,
        targets,
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 25 == 0:
        print(
            f"step={step:3d}, "
            f"loss={loss.item():.4f}"
        )


# 从数字 3 开始生成。
start_tokens = torch.tensor([[3]])

generated = model.generate(
    start_tokens,
    max_new_tokens=12,
)

print("\n生成结果：")
print(generated.tolist())