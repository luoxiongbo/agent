import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def make_data(n_samples=2000):
    # X: [N, 2], 每个样本是一个二维点 (x, y)，范围大约在 [-2, 2]
    """
    torch.rand(n_samples, 2) 会生成 [0, 1) 之间的随机数
    """
    x = torch.rand(n_samples, 2) * 4 - 2

    # 点到原点的距离平方：x^2 + y^2
    radius_squared = x[:, 0] ** 2 + x[:, 1] ** 2

    # 单位圆内是 0，单位圆外是 1
    y = (radius_squared >= 1).long()
    return x, y


class CircleClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # 输入是 2 维点，32 表示第一层隐藏层有 32 个神经元
            # 每个神经元都在学一个自己的线性判断
            # 一个神经元智能画一条直线边界。32个神经元就相当于从32 个角度切这个平面
            nn.Linear(2, 32),
            # 非线性函数，激活函数，把很多“直线切片”组合成更复杂的形状，可以理解为：用很多小折线拼出一个圆
            nn.ReLU(),
            # 低级特征 -> 更复杂的判断
            nn.Linear(32, 32),
            # 激活函数，给神经网络加入非线性功能
            nn.ReLU(),
            # 分类，所以是2
            nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.net(x)


def evaluate(model, data_loader):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            logits = model(x_batch)
            predictions = torch.argmax(logits, dim=1)

            correct += (predictions == y_batch).sum().item()
            total += y_batch.size(0)

    return correct / total


def main():
    torch.manual_seed(0)

    x, y = make_data(n_samples=2000)

    split_index = int(len(x) * 0.8)
    x_train, x_test = x[:split_index], x[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # 把多个 Tensor 按照第 0 维 —— 对应地包装成 “数据集”，输入-输出
    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)

    # 从数据集按批次取数据，并负责打乱、并行加载等
    # 训练集一般都是 shuffle，验证集和测试集一般不用 shuffle
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128)

    # 模型
    model = CircleClassifier()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, 51):
        model.train()

        total_loss = 0.0

        """
        每次循环，data loader根据数据集长度生成索引
        决定索引是否打乱
        每次取出一批索引
        调用 dataset[index]
        把单个样本合并成 batch
        返回一个批次的数据
        
        为什么不是一次把全部数据喂进去？
        1、因为显存可能放不下
        2、模型不需要看完全部数据才更新，每一批更新一次参数
        3、由于一小批数据计算出来的梯度只是整体梯度的近似，所以小批量梯度带有适当的随机性，但是适度的随机性不一定是坏事
            (1)避免模型过早陷入不理想的区域
            (2)提升训练过程中的探索能力
            (3)改善模型在新数据上的表现
        """
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()

            logits = model(x_batch)
            loss = loss_fn(logits, y_batch)

            # 计算每个参数的梯度
            loss.backward()
            # 梯度下降
            # 按照优化算法更新参数
            optimizer.step()

            """
            加上该批的 loss
            """
            total_loss += loss.item() * x_batch.size(0)

        if epoch % 10 == 0:
            avg_loss = total_loss / len(train_dataset)
            test_accuracy = evaluate(model, test_loader)
            print(
                f"epoch={epoch:02d}, "
                f"loss={avg_loss:.4f}, "
                f"test_accuracy={test_accuracy:.2%}"
            )

    torch.save(model.state_dict(), "circle_classifier.pt")

    reloaded_model = CircleClassifier()
    reloaded_model.load_state_dict(torch.load("circle_classifier.pt"))
    reloaded_accuracy = evaluate(reloaded_model, test_loader)
    print(f"reloaded_test_accuracy={reloaded_accuracy:.2%}")

    sample_points = torch.tensor(
        [
            [0.0, 0.0],
            [0.5, 0.5],
            [1.0, 1.0],
            [2.0, 0.0],
        ]
    )
    sample_logits = model(sample_points)
    # 预测类别，logits [圆内分数, 圆外分数] 里分数较高的那个
    sample_predictions = torch.argmax(sample_logits, dim=1)
    print("sample_points:", sample_points)
    print("sample_predictions:", sample_predictions)


if __name__ == "__main__":
    main()
