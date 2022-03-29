import torch
from torch import nn

# 单隐藏层的多层感知机
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size=(2, 4))
net(X)

# 查看参数
print(net[2].state_dict())  # 第二个全连接层
# OrderedDict([('weight', tensor([[-0.3169, -0.1524, -0.1462, -0.1193, -0.2590,  0.1933,  0.1860,  0.2619]])), ('bias', tensor([0.0269]))])
# 提取参数的方法
print(type(net[2].bias))
print(net[2].bias)
print(net[2].bias.data)

# 参数是复合的对象，包含值、梯度和额外信息。
print(net[2].weight.grad == None)

# 一次性访问所有参数
print(*[(name, param.shape) for name, param in net[0].named_parameters()])  # 第一个全连接层 * 表示把列表拆开
print(*[(name, param.shape) for name, param in net.named_parameters()])  # 所有层
print(net.state_dict()['2.bias'].data)
print(net.state_dict())  # 是一个OrderedDict


# 从嵌套块收集参数
def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                         nn.Linear(8, 4), nn.ReLU())


def block2():
    net = nn.Sequential()
    for i in range(4):
        # 在这里嵌套
        net.add_module(f'block {i}', block1())
    return net


rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
print(rgnet(X))
# 查看网络
print(rgnet)
print(rgnet[0][1][0].bias.data)


# 内置初始化
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)


net.apply(init_normal)
print(net[0].weight.data[0], net[0].bias.data[0])


# 或者初始化为常数
def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)


net.apply(init_constant)
print(net[0].weight.data[0], net[0].bias.data[0])


# 自定义初始化
def my_init(m):
    if type(m) == nn.Linear:
        print("Init", *[(name, param.shape)
                        for name, param in m.named_parameters()][0])
        nn.init.uniform_(m.weight, -10, 10)
        m.weight.data *= m.weight.data.abs() >= 5


net.apply(my_init)
print(net[0].weight[:2])


# 参数绑定
# 有时我们希望在多个层间共享参数： 我们可以定义一个稠密层，然后使用它的参数来设置另一个层的参数。
# 我们需要给共享层一个名称，以便可以引用它的参数
shared = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                    shared, nn.ReLU(),
                    shared, nn.ReLU(),
                    nn.Linear(8, 1))
net(X)
# 检查参数是否相同
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
# 确保它们实际上是同一个对象，而不只是有相同的值
print(net[2].weight.data[0] == net[4].weight.data[0])
