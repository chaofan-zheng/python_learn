import torch
import numpy as np
from matplotlib import pyplot as plt
import math


def test1():
    """
    指定一个函数，自动计算梯度
    :return:
    """
    x = torch.arange(4.0)
    print(x)

    x.requires_grad_(True)  # 等价于x=torch.arange(4.0,requires_grad=True) 表示需要求导
    print(x.grad)  # 默认是None

    y = 2 * torch.dot(x, x)  # 假设我们想对函数 𝑦=2𝐱⊤𝐱 关于列向量 𝐱 求导。
    print(y)  # tensor(28., grad_fn=<MulBackward0>)

    # 反向传播函数来自动计算y关于x每个分量的梯度，并打印这些梯度。
    y.backward()
    print(x.grad)
    # 验证梯度计算是否正确
    print(x.grad == 4 * x)

    # 在默认情况下，PyTorch会累积梯度，我们需要清除之前的值
    x.grad.zero_()


def test2():
    x = torch.arange(4.0)
    x.requires_grad_(True)
    y = x.sum()
    y.backward()
    print(x.grad)


def test3():
    """
    非标量变量的反向传播
    :return:
    """
    x = torch.arange(4.0)
    x.requires_grad_(True)
    # 对非标量调用backward需要传入一个gradient参数，该参数指定微分函数关于self的梯度。
    # 在我们的例子中，我们只想求偏导数的和，所以传递一个1的梯度是合适的
    y = x * x
    # 等价于y.backward(torch.ones(len(x)))
    y.sum().backward()
    print(x.grad)


def test4():
    """
    分离计算
    求 z关于x的梯度，但由于某种原因，我们希望将y视为一个常数， 并且只考虑到x在y被计算后发挥的作用。
    此时需要把y视为一个常数u
    :return:
    """
    x = torch.arange(4.0, requires_grad=True)
    y = x * x
    u = y.detach()  # 但丢弃计算图中如何计算y的任何信息
    z = u * x

    z.sum().backward()
    #
    print(x.grad == u)

    x.grad.zero_()
    y.sum().backward()  # 由于记录了y的计算结果，我们可以随后在y上调用反向传播， 得到y=x*x关于的x的导数，即2*x。
    print(x.grad == 2 * x)


def f(a):
    """
    python 控制流的梯度计算
    :param a:
    :return:
    """
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c


def test5():
    a = torch.randn(size=(), requires_grad=True)  # 随机数
    d = f(a)
    d.backward()
    print(a.grad == d / a)


def test6():
    a = torch.randn(size=(3, 4), requires_grad=True)  # 随机数 3，4的矩阵
    d = f(a)
    d.sum().backward()
    print(a.grad == d / a)


def f2(x: torch.tensor):
    # y = torch.tensor([math.sin(i) for i in x])
    y = torch.sin(x)
    return y


def test7():
    """
    使 𝑓(𝑥)=sin(𝑥) ，绘制 𝑓(𝑥) 和 𝑑𝑓(𝑥)𝑑𝑥 的图像，其中后者不使用 𝑓′(𝑥)=cos(𝑥) 。
    :return:
    """
    x = np.linspace(-3 * np.pi, 3 * np.pi, 100)
    x1 = torch.tensor(x, requires_grad=True)
    # 带有requires_grad=True 需要求导的张量不能够画图
    y1 = f2(x1)
    print(y1)
    y1.sum().backward()

    plt.plot(x, y1.detach().numpy(), label="sinx") # 带有requires_grad=True 需要求导的张量不能够画图
    plt.plot(x, x1.grad, label="cosx")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # test1()
    test7()
