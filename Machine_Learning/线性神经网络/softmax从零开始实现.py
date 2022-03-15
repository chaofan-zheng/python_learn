import torch
from IPython import display
from d2l import torch as d2l

def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)  # 对数据每一行求和
    return X_exp / partition  # 这里应用了广播机制

def net(X):
    """
    softmax回归模型
    注意，将数据传递到模型之前，我们使用reshape函数将每张原始图像展平为向量。
    :param X:
    :return:
    """
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)

def cross_entropy(y_hat, y):
    """
    定义交叉熵损失函数

    :param y_hat: 预测值概率 y_hat[range(len(y_hat)), y] 获得目标预测值的概率
    :param y:
    :return:
    """
    return - torch.log(y_hat[range(len(y_hat)), y])

if __name__ == '__main__':

    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    # 在本例中 我们将28*28的数据平展为长度784的向量 我们将讨论能够利用图像空间结构的特征，但现在我们暂时只把每个像素位置看作一个特征。
    # 回想一下，在softmax回归中，我们的输出与类别一样多。
    # 因为我们的数据集有10个类别，所以网络输出维度为10。 因此，权重将构成一个的矩阵， 偏置将构成一个的行向量。
    # 与线性回归一样，我们将使用正态分布初始化我们的权重W，偏置初始化为0。
    num_inputs = 784
    num_outputs = 10
    W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
    b = torch.zeros(num_outputs, requires_grad=True)

