import torch
from torch import nn
from d2l import torch as d2l
from tools.trainer import basic_trainer, show_images
import os

os.environ["OMP_NUM_THREADS"] = "1"


def relu(X):
    """
    In [13]: X
    Out[13]:
    tensor([[ 1.,  2.],
            [-1.,  0.]])

    In [14]: relu(X)
    Out[14]:
    tensor([[1., 2.],
            [0., 0.]])
    :param X:
    :return:
    """
    a = torch.zeros_like(X)
    return torch.max(X, a)


def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(X.matmul(W1) + b1)  # 隐藏层
    return H.matmul(W2) + b2
    # H = relu(X@W1 + b1)  # 这里“@”代表矩阵乘法
    # return (H@W2 + b2)


def predict_ch3(net, test_iter, n=6):  # @save
    """预测标签（定义见第3章）"""
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
    show_images(X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])


if __name__ == '__main__':
    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    # 单隐藏层的多层感知机， 它包含256个隐藏单元。
    num_inputs, num_outputs, num_hiddens = 784, 10, 256  # 28*28 = 784，一共有10个类别，256个隐藏单元，
    # 我们选择2的若干次幂作为层的宽度。 因为内存在硬件中的分配和寻址方式，这么做往往可以在计算上更高效。
    # 初始化模型参数
    W1 = nn.Parameter(torch.randn(
        num_inputs, num_hiddens, requires_grad=True) * 0.01)
    b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
    W2 = nn.Parameter(torch.randn(
        num_hiddens, num_outputs, requires_grad=True) * 0.01)
    b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

    params = [W1, b1, W2, b2]
    loss = nn.CrossEntropyLoss(reduction='none')

    # 训练
    num_epochs, lr = 10, 0.1
    updater = torch.optim.SGD(params, lr=lr)
    # d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
    basic_trainer(net, train_iter, test_iter, loss, num_epochs, updater)

    predict_ch3(net, test_iter)
