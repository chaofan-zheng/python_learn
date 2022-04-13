"""
输入Xt-1,..x1 本身因t而异。也就是说输入数据的数量会随着事件改变。如何用一个近似方法使得计算变得容易？
1. 自回归
    使用一个周期T，使用观测序列Xt-1,..xt-T
2. 隐变量自回归
    保留一些对于过去观测的总结ht，并且同时更新预测xt和总结ht

"""

# 生成数据
import torch
from torch import nn
from d2l import torch as d2l
from matplotlib import pyplot as plt

T = 1000  # 总共产生1000个点
time = torch.arange(1, T + 1, dtype=torch.float32)
x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))
d2l.plot(time, [x], xlim=[1, 1000], figsize=(6, 3))
plt.show()

#  构建数据迭代器
tau = 4
features = torch.zeros((T - tau, tau))
for i in range(tau):
    features[:, i] = x[i: T - tau + i]
labels = x[tau:].reshape((-1, 1))

batch_size, n_train = 16, 600  # 只使用600个标签对进行训练
# 只有前n_train个样本用于训练
train_iter = d2l.load_array((features[:n_train], labels[:n_train]),
                            batch_size, is_train=True)


# 初始化网络权重的函数
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)


# 一个简单的多层感知机
def get_net():
    net = nn.Sequential(nn.Linear(4, 10),
                        nn.ReLU(),
                        nn.Linear(10, 1))
    net.apply(init_weights)
    return net


# 平方损失。注意：MSELoss计算平方误差时不带系数1/2
loss = nn.MSELoss(reduction='none')


def train(net, train_iter, loss, epochs, lr):
    trainer = torch.optim.Adam(net.parameters(), lr)
    for epoch in range(epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.sum().backward()
            trainer.step()
        print(f'epoch {epoch + 1}, '
              f'loss: {d2l.evaluate_loss(net, train_iter, loss):f}')


net = get_net()
train(net, train_iter, loss, 5, 0.01)

# 预测，单次预测能够有很好的工作效果
onestep_preds = net(features)
figsize = (6, 3)
plt.plot(time, x.detach().numpy())

# k步预测，对于观察到xt的序列，在时间步xt+k的预测，就是k步预测
multistep_preds = torch.zeros(T)
multistep_preds[: n_train + tau] = x[: n_train + tau]
for i in range(n_train + tau, T):
    multistep_preds[i] = net(
        multistep_preds[i - tau:i].reshape((1, -1)))
plt.plot(time[n_train + tau:],multistep_preds[n_train + tau:].detach().numpy())
# 经过几个预测步骤之后，预测的结果很快就会衰减到一个常数。 为什么这个算法效果这么差呢？事实是由于错误的累积：

plt.show()