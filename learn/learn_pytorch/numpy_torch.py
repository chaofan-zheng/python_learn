"""
Torch 是神经网络界的Numpy，他能将torch产生的tensor放在GPU中加速运算。
Numpy会把array放在CPU中加速运算
"""
import torch
import numpy as np

# numpy 和 tensor进行一些转化
np_data = np.arange(6).reshape((2, 3))
torch_data = torch.from_numpy(np_data)
tensor2array = torch_data.numpy()

# abs 绝对值
data = [-1, -2, 1, 2]
tensor_data = torch.FloatTensor(data)
print(torch.abs(tensor_data))
print(torch.mean(tensor_data))
# 计算符查询
# https://pytorch.org/docs/stable/torch.html#math-operations

# 矩阵运算
data = [[1, 2], [3, 4]]
tensor_data = torch.FloatTensor(data)
data = np.array(data)
print(
    "\nNumpy:", np.matmul(data, data),
    "\nNumpy:", data.dot(data),
    "\ntorch:", torch.mm(tensor_data, tensor_data),  # 矩阵相乘
    # "\ntorch:", tensor_data.dot(tensor_data),  # 报错，不太一样
    "\ntorch:", torch.dot(torch.tensor([2, 3]), torch.tensor([2, 1])),  # 输入必须是1D
    "\ntorch:", torch.matmul(torch.tensor([2, 3]), torch.tensor([2, 1])),
    "\ntorch:", torch.matmul(torch.tensor([1, 2]), torch.tensor([[1, 2], [3, 4]])),  # torch: tensor([ 7, 10])
    "\ntorch:", torch.matmul(torch.tensor([[1, 2], [3, 4], [5, 6]]), torch.tensor([1, 2])), # torch: tensor([ 5, 11, 17])
    """
    matmul 的行为取决于 张量的维度
    
    如果两个张量都是一维的，则返回点积（标量）。

    如果两个参数都是二维的，则返回矩阵-矩阵乘积。

    如果第一个参数是一维的，第二个参数是二维的，为了矩阵乘法的目的，在它的维数前面加上一个 1。在矩阵相乘之后，前置维度被移除。

    如果第一个参数是二维的，第二个参数是一维的，则返回矩阵向量积。
    
    """
)
