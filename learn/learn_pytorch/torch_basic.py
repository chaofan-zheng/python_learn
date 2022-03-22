import torch


def test1():
    """
    创建一些张量
    :return:
    """
    print(torch.Tensor([1, 2, 3, 4, 5]))
    t = torch.Tensor(2, 3)  # shape 为 2，3 随机初始化
    print(t)
    print(t.size())
    print(t.shape)
    t1 = torch.Tensor(1)
    t2 = torch.tensor(1)
    print(f"torch.Tensor :{t1}, 数据类型：{t1.type()}")
    print(f"torch.tensor :{t2}, 数据类型：{t2.type()}")
    # torch.Tensor 与 torch.tensor 的区别
    # torch.Tensor 是 torch.empty 和 torch.tensor 的一种混合，指定的是shape，进行随机初始化
    # torch.tensor 指定值，从数据中推断类型

    print(torch.eye(2, 2))
    print(torch.zeros(2, 3))
    print(torch.linspace(1, 10, 4))  # 左闭右闭，步长4
    print(torch.rand(2, 3))  # 均匀分布随机数
    print(torch.randn(2, 3))  # 标准分布随机数
    print(torch.zeros_like(torch.rand(2, 3)))  # 形状相同，全0


def test2():
    """
    修改张量形状
    :return:
    """
    x = torch.randn(2, 3)
    print(x.size())
    print(x.dim())  # 查看x的维度


if __name__ == '__main__':
    test1()
