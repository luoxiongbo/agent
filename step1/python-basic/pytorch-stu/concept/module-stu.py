import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    # 定义层
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(in_features=2, out_features=1)

    # 定义数据怎么流动,前向传播
    def forward(self, x):
        return self.linear(x)


model = SimpleModel()

x = torch.tensor([[1.0, 2.0]])
# pytorch 会自动去调用 forward
y_pred = model(x)
print(y_pred)
