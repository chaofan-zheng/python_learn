"""
总结：
X = x.reshape(-1, 4)  # -1 自动计算
print(np.random.normal(0, 1, size=(3, 4)))  # 其中的每个元素都从均值为0、标准差为1的标准高斯分布（正态分布）中随机采样。
print(np.concatenate([X, Y], axis=0)) # 两表拼接
Z[:] = X + Y 节省内存
广播：通过适当复制元素来扩展一个或两个数组， 以便在转换之后，两个张量具有相同的形状。
"""

import numpy as np

x = np.arange(12)
print(x)

print(x.shape)
print(x.size)  # shape的乘积
X = x.reshape(-1, 4)  # -1 自动计算
print(X)
print(np.zeros((2, 3, 4)))
print(np.random.normal(0, 1, size=(3, 4)))  # 其中的每个元素都从均值为0、标准差为1的标准高斯分布（正态分布）中随机采样。

# 连接张量 concatenate 我们只需要提供张量列表，并给出沿哪个轴连结。
X = np.arange(12).reshape(3, 4)
Y = np.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print(np.concatenate([X, Y], axis=0))
print(np.concatenate([X, Y], axis=1))

print(X == Y)  # XY的shape要相同，比较每个单元格的是否相同，返回相同size的01张量
print(X.sum())  # 对张量的所有元素进行求和

"""
广播机制
在某些情况下，即使形状不同，我们仍然可以通过调用 广播机制（broadcasting mechanism）来执行按元素操作。
这种机制的工作方式如下：首先，通过适当复制元素来扩展一个或两个数组， 以便在转换之后，两个张量具有相同的形状。 
其次，对生成的数组执行按元素操作。
"""
a = np.arange(3).reshape(3, 1)
b = np.arange(2).reshape(1, 2)
print(a)
print(b)
print(a + b)  # 广播成3，2 矩阵a将复制列， 矩阵b将复制行，然后再按元素相加。

# 索引和切片
print(X[-1])  # 获取最后一行
print(X[1:3])  # 获取 1到3行 左闭右开
X[0:2, :] = 12  # 对1~2行的所有列进行赋值操作
print(X)

# 节省内存
before = id(Y)  # 获取地址
Y = Y + X
print(id(Y) == before)  # Flase 修改了变量之后还是使用了新的内存
# 我们希望原地进行更改
Z = np.zeros_like(Y)  # 和Y有相同形状的全0张量
print('id(Z):', id(Z))
Z[:] = X + Y
print('id(Z):', id(Z))
