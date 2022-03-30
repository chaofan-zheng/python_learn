import torch
from torch import nn
from d2l import torch as d2l
from tools.trainer import Animator

# 输入28*28
net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=3, padding=2), nn.ReLU(),  # 特征图大小 = (28-5+2*2+1) = 28
    nn.AvgPool2d(kernel_size=2, stride=2),  # 特征图大小 = (28-2+1+2)/2=14
    nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(),  # (14 - 5 +1) = 10
    nn.AvgPool2d(kernel_size=2, stride=2),  # 特征图大小 = (10-2+2+1)/2
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120), nn.ReLU(),  # 输入
    nn.Linear(120, 84), nn.ReLU(),
    nn.Linear(84, 10))

# 查看输出形状大小
X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape: \t', X.shape)

# 训练LNet网络
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)


def evaluate_accuracy_gpu(net, data_iter, device=None):  # @save
    """使用GPU计算模型在数据集上的精度"""
    if isinstance(net, nn.Module):
        net.eval()  # 设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device
    # 正确预测的数量，总预测的数量
    metric = d2l.Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # BERT微调所需的（之后将介绍）
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(d2l.accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


# @save
def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    """Train a model with a GPU (defined in Chapter 6).

    Defined in :numref:`sec_lenet`"""
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = d2l.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples
        metric = d2l.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')
    animator.show()


lr, num_epochs = 0.9, 10
# train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())

# 优化网络
"""
1. 修改卷积核 5->3
2. 激活函数替换成ReLU # 更快收敛
3. 增大输出通道
4. 增加卷积层数量
5. 增加全连接层
5. 修改学习率
7. 修改轮数
"""
net = nn.Sequential(
    nn.Conv2d(1, 5, kernel_size=7, padding=3), nn.ReLU(),  # 特征图大小 = (28-7+3*2+1) = 28 调整了卷积核大小
    nn.MaxPool2d(kernel_size=2, stride=2),  # 特征图大小 = (28-2+2)/2=14
    nn.Conv2d(5, 10, kernel_size=5, padding=2), nn.ReLU(),  # (14 - 5 +1+2*2) = 14
    nn.MaxPool2d(kernel_size=2, stride=2),  # 特征图大小 = (14-2+2)/2 = 7
    nn.Conv2d(10, 20, kernel_size=3), nn.ReLU(),  # (7 - 3 +1) = 5
    nn.MaxPool2d(kernel_size=2,padding=1),  # 特征图大小 = (5-2+2+1)/1 = 5
    nn.Flatten(),
    nn.Linear(20 * 3 * 3, 120), nn.ReLU(),
    nn.Linear(120, 84), nn.ReLU(),
    nn.Linear(84, 50), nn.ReLU(),
    nn.Linear(50, 10))
X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape: \t', X.shape)
lr, num_epochs = 0.01, 10
train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
