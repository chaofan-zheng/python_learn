import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l
from torch import nn


def load_array(data_arrays, batch_size, is_train=True):  # @save
    """构造一个PyTorch数据迭代器 随机、小批量、不重复"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


if __name__ == '__main__':
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = d2l.synthetic_data(true_w, true_b, 1000)

    batch_size = 10
    data_iter = load_array((features, labels), batch_size)
    print(next(iter(data_iter)))

    # 定义模型
    net = nn.Sequential(nn.Linear(2, 1))  # Sequential类的实例
    # Sequential类将多个层串联在一起。 当给定输入数据时，Sequential实例将数据传入到第一层， 然后将第一层的输出作为第二层的输入
    # 在PyTorch中，全连接层在Linear类中定义。 值得注意的是，我们将两个参数传递到nn.Linear中
    # 一个指定输入特征形状，即2，第二个指定输出特征形状，输出特征形状为单个标量，因此为1。

    net[0].weight.data.normal_(0, 0.01)  # net[0]网络的第一个图层
    net[0].bias.data.fill_(0)  # 然后使用weight.data和bias.data方法访问参数。 我们还可以使用替换方法normal_和fill_来重写参数值。

    # 定义损失函数
    loss = nn.MSELoss()  # 平方范数

    # 定义优化算法
    trainer = torch.optim.SGD(net.parameters(), lr=0.03)

    # 训练
    num_epochs = 3
    for epoch in range(num_epochs):
        for X, y in data_iter:
            l = loss(net(X), y)
            trainer.zero_grad()
            l.backward()
            trainer.step() # 使用参数的梯度更新参数
        l = loss(net(features), labels)
        print(f'epoch {epoch + 1}, loss {l:f}')
