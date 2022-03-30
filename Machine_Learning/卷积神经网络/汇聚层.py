import torch
from torch import nn


def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i: i + p_h, j: j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()
    return Y


if __name__ == '__main__':
    X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
    print(pool2d(X, (2, 2), mode='avg'))

    X = torch.arange(16, dtype=torch.float32).reshape((1, 1, 4, 4))
    print(X)

    pool2d = nn.MaxPool2d(3)  # 默认情况下，深度学习框架中的步幅与汇聚窗口的大小相同  因此，如果我们使用形状为(3, 3)的汇聚窗口，那么默认情况下，我们得到的步幅形状为(3, 3)。
    print(pool2d(X))

    # 自定义padding和stride
    pool2d = nn.AvgPool2d(3, padding=1, stride=2) # 每边都填充了1，所以高度和宽度都是1
    print(pool2d(X))

    # 矩形
    pool2d = nn.MaxPool2d((2, 3), stride=(2, 3), padding=(0, 1))
    pool2d(X)

    # 当数据有多个通道的时候，汇聚层在每个输入通道上单独运算，而不是像卷积层一样在通道上对输入进行汇总。 这意味着汇聚层的输出通道数与输入通道数相同

