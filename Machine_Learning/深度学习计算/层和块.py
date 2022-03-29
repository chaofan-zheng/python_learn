import torch
from torch import nn
from torch.nn import functional as F

"""
自定义块

1. 将输入数据作为其前向传播函数的参数。

2. 通过前向传播函数来生成输出。请注意，输出的形状可能与输入的形状不同。例如，我们上面模型中的第一个全连接的层接收一个20维的输入，但是返回一个维度为256的输出。

3. 计算其输出关于输入的梯度，可通过其反向传播函数进行访问。通常这是自动发生的。

4. 存储和访问前向传播计算所需的参数。

5. 根据需要初始化模型参数。
"""


class MLP(nn.Module):
    # 用模型参数声明层。这里，我们声明两个全连接的层
    def __init__(self):
        # 调用MLP的父类Module的构造函数来执行必要的初始化。
        # 这样，在类实例化时也可以指定其他函数参数，例如模型参数params（稍后将介绍）
        super().__init__()
        self.hidden = nn.Linear(20, 256)  # 隐藏层(新)
        self.out = nn.Linear(256, 10)  # 输出层

    # 定义模型的前向传播，即如何根据输入X返回所需的模型输出
    def forward(self, X):
        # 注意，这里我们使用ReLU的函数版本，其在nn.functional模块中定义。
        return self.out(F.relu(self.hidden(X)))


class MySequential(nn.Module):
    def __init__(self, *args: nn.Module):
        super().__init__()
        for idx, module in enumerate(args):
            # 这里，module是Module子类的一个实例。我们把它保存在'Module'类的成员
            # 变量_modules中。module的类型是OrderedDict 使用OrderedDict会根据放入元素的先后顺序进行排序
            self._modules[str(idx)] = module
        # 你可能会好奇为什么每个Module都有一个_modules属性？ 以及为什么我们使用它而不是自己定义一个Python列表？
        # 简而言之，_modules的主要优点是： 在模块的参数初始化过程中， 系统知道在_modules字典中查找需要初始化参数的子块。
    def forward(self, X):
        # OrderedDict保证了按照成员添加的顺序遍历它们
        for block in self._modules.values():
            X = block(X)
        return X

# 自定义前向传播
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 不计算梯度的随机权重参数。因此其在训练期间保持不变
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)

    def forward(self, X):
        X = self.linear(X)
        # 使用创建的常量参数以及relu和mm函数
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        # 实现了一个隐藏层， 其权重（self.rand_weight）在实例化时被随机初始化，之后为常量。 这个权重不是一个模型参数，因此它永远不会被反向传播更新。 然后，神经网络将这个固定层的输出通过一个全连接层。

        # 复用全连接层。这相当于两个全连接层共享参数
        X = self.linear(X)
        # 控制流
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()

if __name__ == '__main__':
    #  下面的代码生成一个网络，其中包含一个具有256个单元和ReLU激活函数的全连接隐藏层， 然后是一个具有10个隐藏单元且不带激活函数的全连接输出层。
    net = nn.Sequential(nn.Linear(20, 256),  # 输入20个特征，输出256个隐藏层单元
                        nn.ReLU(),
                        nn.Linear(256, 10))  # 输入256个特征，输出10个

    X = torch.rand(2, 20)
    print(net(X))
    """
    简而言之，nn.Sequential定义了一种特殊的Module， 即在PyTorch中表示一个块的类， 它维护了一个由Module组成的有序列表。 
    注意，两个全连接层都是Linear类的实例， Linear类本身就是Module的子类。 
    另外，到目前为止，我们一直在通过net(X)调用我们的模型来获得模型的输出。 这实际上是net.__call__(X)的简写。 
    这个前向传播函数非常简单： 它将列表中的每个块连接在一起，将每个块的输出作为下一个块的输入。
    """
    # 测试MLP
    net = MLP()
    print(net(X))
