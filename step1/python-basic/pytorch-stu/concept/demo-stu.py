import torch
import torch.nn as nn

"""
让模型学 y = 2x
"""

x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y = torch.tensor([[2.0], [4.0], [6.0], [8.0]])

"""
in_features: 输入特征数
out_features: 输出特征数
"""
model = nn.Linear(1, 1)

"""
创建一个均方误差损失函数（MSE：Mean Squared Error）
用来衡量模型预测值和真实值差的有多远
公式大概为：MSE = 平均值((预测值 - 真实值)^2)
loss 越小，说明预测越准
常用于 回归任务，比如预测价格、温度、销量、坐标这类连续数值。
"""
loss_fn = nn.MSELoss()

"""
model.parameters() 里 p.grad 保存着每个参数的梯度
lr 学习率
SGD 算法，大致原理：新参数 = 旧参数 - 学习率 * 梯度
比如：
w = 3.0
w.grad = 2（梯度的确是2）
lr = 0.01（学习率大会过拟合？）
新 w = 3.0 - 0.01 * 2 = 2.98
让损失 loss 下降
"""
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(1000):
    """
    清梯度 -> 前向预测 -> 计算损失 -> 反向求梯度 -> 更新参数
    """

    # 预测
    """
    内部，pytorch 会自动去调用 forward()
    只要模型参数默认是 requires_grad=True，PyTorch 就会记录：
    参数 -> 中间计算 -> 预测值 -> loss
    """
    y_pred = model(x)

    # 计算损失
    """
    预测值与实际值之间的损失
    """
    loss = loss_fn(y_pred, y)

    # 清空梯度
    """
    pytorch 默认会累积梯度
    梯度会不断累加，而不是自动覆盖
    清空梯度把上一轮保留下的 .grad 清掉，保证这一轮的梯度只来自当前 batch
    该行建议写在最前面，习惯！
    """
    optimizer.zero_grad()

    # 反向传播
    """
    PyTorch 从 loss 开始，沿着计算图反向走，用链式法则计算每个参数对 loss 的影响。
    结果会存到每个参数的 梯度 .grad 里。
    """
    loss.backward()

    # 更新参数
    """
    根据优化算法更新参数
    """
    optimizer.step()

    if epoch % 10 == 0:
        print(epoch, loss.item())

print(x.shape)
print(torch.tensor([[5.0]]).shape)
print(model(torch.tensor([[5.0]])))
print("weight:", model.weight.data)
print("bias:", model.bias.data)
