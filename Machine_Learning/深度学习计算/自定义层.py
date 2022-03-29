import torch
import torch.nn.functional as F
from torch import nn


# 不带参数的层
class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()


# 带参数的自定义层
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))  # 输入数，输出数
        self.bias = nn.Parameter(torch.randn(units, ))

    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)


if __name__ == '__main__':
    # 作为组件放到更复杂的层当中
    net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
