import torch

# requires_grad=True 表示：我要对这个变量求梯度
w = torch.tensor(2.0, requires_grad=True)

# 假设损失函数是 y = w^2
loss = w ** 2

# 反向传播，计算 d(loss)/d(w)
loss.backward()

# 2 * 2.0 = 4.0
print(w.grad)