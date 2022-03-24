import torch
from d2l import torch as d2l
from matplotlib import pyplot as plt


def test1():
    """梯度消失"""
    x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
    y = torch.sigmoid(x)

    y.backward(torch.ones_like(x))  # y 必须要是一个标量（shape为1的张量，），如果不是需要指定grad_tensors,对grad_tensors进行点乘
    # 梯度消失
    plt.plot(x.detach().numpy(), y.detach().numpy(), label="sigmoid")
    plt.plot(x.detach().numpy(), x.grad.numpy(), label="gradient")
    plt.legend()
    plt.show()
    # 当sigmoid函数的输入很大或是很小时，它的梯度都会消失。 此外，当反向传播通过许多层时，除非我们在刚刚好的地方， 这些地方sigmoid函数的输入接近于零，否则整个乘积的梯度可能会消失。


def test2():
    """梯度爆炸"""
    # 梯度爆炸
    M = torch.normal(0, 1, size=(4, 4))
    print('一个矩阵 \n', M)
    for i in range(100):
        M = torch.mm(M, torch.normal(0, 1, size=(4, 4)))

    print('乘以100个矩阵后\n', M)


if __name__ == '__main__':
    test2()
