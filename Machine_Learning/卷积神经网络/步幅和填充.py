import torch
from torch import nn


# 为了方便起见，我们定义了一个计算卷积层的函数。
# 此函数初始化卷积层权重，并对输入和输出提高和缩减相应的维数
def comp_conv2d(conv2d, X):
    # 这里的（1，1）表示批量大小和通道数都是1
    X = X.reshape((1, 1) + X.shape)
    print(X.shape)
    Y = conv2d(X)
    # 省略前两个维度：批量大小和通道
    return Y.reshape(Y.shape[2:])


def padding():
    # 请注意，这里每边都填充了1行或1列，因此总共添加了2行或2列
    conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1)
    X = torch.rand(size=(8, 8))
    print(comp_conv2d(conv2d, X).shape)

    # 当卷积核的高度和宽度不同时，我们可以填充不同的高度和宽度，高度和宽度两边的填充分别为2和1。
    conv2d = nn.Conv2d(1, 1, kernel_size=(5, 3), padding=(2, 1))
    print(comp_conv2d(conv2d, X).shape)


def stride():
    conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)
    X = torch.rand(size=(8, 8))
    print(comp_conv2d(conv2d, X).shape)


if __name__ == '__main__':
    padding()
    stride()
