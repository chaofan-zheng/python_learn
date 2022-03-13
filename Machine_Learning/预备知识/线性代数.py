"""
总结：
1. 降维求和 求平均
    A_sum_axis0 = A.sum(axis=0) 维度变小
    print(A.mean(axis=0))
2. 保持维度不变求和求平均
    sum_A = A.sum(axis=1, keepdims=True)
3. 保持维度求和 在通过广播，可以求得在各个行、列中的比例
    # 非降维求和
    sum_A = A.sum(axis=1, keepdims=True)
    print(sum_A)
    # 通过广播求比例
    print(A / sum_A)
4. 范数：用来告诉我们张量有多大（类似于向量的模的概念）
    L1范数：绝对值的和
    L2范数：欧氏距离，也就是平方和开根号
"""

import numpy as np

A = np.arange(20).reshape(5, 4)
print(A.T)  # 矩阵的转置

# 降维
print(A)
A_sum_axis0 = A.sum(axis=0)
print(A_sum_axis0)
print(A_sum_axis0.shape)
A_sum_axis1 = A.sum(axis=1)
print(A_sum_axis1)
print(A_sum_axis1.shape)
print(A.sum(axis=(0, 1)))  # 190

# mean 平均值 也是跟sum一样可以指定维度
print(A.mean(), A.sum() / A.size)
print(A.mean(axis=0))

# 非降维求和
sum_A = A.sum(axis=1, keepdims=True)
print(sum_A)
# 通过广播求比例
print(A / sum_A)
print(A.cumsum(axis=0)) # 按行计算，不会降低维度

# 点积
x = np.arange(4)
y = np.ones(4)
print(x)
print(y)
print(np.dot(x, y))  # xTy

# 向量积
print(A.shape, x.shape)
print(np.dot(A, x))

# 矩阵相乘
B = np.ones(shape=(4, 3))
print(np.dot(A, B))

# 范数：
print(np.linalg.norm(np.ones((4, 9))))