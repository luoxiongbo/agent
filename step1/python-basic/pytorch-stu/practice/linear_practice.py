import torch
import torch.nn as nn

torch.manual_seed(0)

"""
实际上是 x1，x2
"""
x = torch.randn(1000, 2)
y = 3 * x[:, 0:1] - 2 * x[:, 1:2] + 5

# print(x)
# print(y)

model = nn.Linear(2, 1)
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 1000 次直接到位了
for epoch in range(1000):
    # 预测
    y_pred = model(x)

    # 计算损失
    loss = loss_fn(y_pred, y)

    # 清空梯度
    optimizer.zero_grad()

    # 反向传播
    loss.backward()

    # 优化参数
    optimizer.step()

    if epoch % 10 == 0:
        print(epoch, loss.item())

print("🧩x's shape:", x.shape)
print("🧩input's shape", torch.tensor([[1.0, 2.0]]).shape)

print("📈predicate_result:", model(torch.tensor([[1.0, 2.0]])))
print("weight:", model.weight)
print("bias:", model.bias)
