import torch
from d2l import torch as d2l


def corr2d_multi_in(X, K):
    # 先遍历“X”和“K”的第0个维度（通道维度），再把它们加在一起
    return sum(d2l.corr2d(x, k) for x, k in zip(X, K))


def corr2d_multi_in_out(X, K):
    # 迭代“K”的第0个维度，每次都对输入“X”执行互相关运算。
    # 最后将所有结果都叠加在一起
    #    for k in K:
    #     print(k.shape) # torch.Size([2, 2, 2]),迭代三次
    return torch.stack([corr2d_multi_in(X, k) for k in K], 0)  # 叠加


def corr2d_multi_in_out_1x1(X, K):
    """
    1x1的卷积核
    :param X:
    :param K:
    :return:
    """
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.reshape((c_i, h * w))
    K = K.reshape((c_o, c_i))
    # 全连接层中的矩阵乘法
    Y = torch.matmul(K, X)
    return Y.reshape((c_o, h, w))


if __name__ == '__main__':
    X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
                      [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
    K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])
    print(corr2d_multi_in(X, K))

    # torch.stack的用法
    print(K.shape)  # torch.Size([2, 2, 2])
    K = torch.stack((K, K + 1, K + 2), 0)  # 在dim = 0 叠加
    print(K.shape)  # torch.Size([3, 2, 2, 2])

    # 多通道运算
    res = corr2d_multi_in_out(X, K)
    print(res)

    # 1x1卷积
    X = torch.normal(0, 1, (3, 3, 3))
    K = torch.normal(0, 1, (2, 3, 1, 1))  # 具有3个输入通道和2个输出通道

    Y1 = corr2d_multi_in_out_1x1(X, K)
    print(Y1)
