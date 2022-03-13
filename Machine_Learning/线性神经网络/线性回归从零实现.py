import random
import torch
from matplotlib import pyplot as plt

"""
我们将根据带有噪声的线性模型构造一个人造数据集。 我们的任务是使用这个有限样本的数据集来恢复这个模型的参数。 
我们将使用低维数据，这样可以很容易地将其可视化。
 在下面的代码中，我们生成一个包含1000个样本的数据集， 每个样本包含从标准正态分布中采样的2个特征。 我们的合成数据集是一个矩阵 𝐗∈ℝ1000×2 。
"""


def synthetic_data(w, b, num_examples):  # @save
    """生成y=Xw+b+噪声"""
    X = torch.normal(0, 1, (num_examples, len(w)))  # 正态分布 mean 为 0, 标准方差为1 ,size
    print(X.shape)  # torch.Size([1000, 2])
    y = torch.matmul(X, w) + b  # 矩阵运算 X*w
    print(y.shape)  # torch.Size([1000])
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))


def data_iter(batch_size, features, labels):
    """ 随机获得小批量数据 """
    """ 随机、不重复"""
    num_examples = len(features)
    indices = list(range(num_examples))
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)  # 随机打乱
    for i in range(0, num_examples, batch_size):  # 按照batch size为步长去取
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]  # 取到每次是随机的并且不重复的


def linreg(X, w, b):
    """线性回归模型"""
    return torch.matmul(X, w) + b


def squared_loss(y_hat, y):
    """定义损失函数：均方损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

def sgd(params, lr, batch_size):
    """
    定义优化算法：小批量随机梯度下降
    :param params: 
    :param lr:
    :param batch_size:
    :return:
    """
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

if __name__ == '__main__':
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = synthetic_data(true_w, true_b, 1000)  # features中的每一行都包含一个二维数据样本， labels中的每一行都包含一维标签值（一个标量）。
    plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1)
    # 通过生成第二个特征features[:, 1]和labels的散点图， 可以直观观察到两者之间的线性关系。
    plt.show()

    # 初始化模型参数
    w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    """
    在初始化参数之后，我们的任务是更新这些参数，直到这些参数足够拟合我们的数据。 
    每次更新都需要计算损失函数关于模型参数的梯度。 有了这个梯度，我们就可以向减小损失的方向更新每个参数。
    """
