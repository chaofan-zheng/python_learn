import torch
from torch.distributions import multinomial
from matplotlib import pyplot as plt


def test1():
    """
    我们只需传入一个概率向量。 输出是另一个相同长度的向量：它在索引 𝑖 处的值是采样结果中 𝑖 出现的次数。
    :return:
    """
    # 掷筛子
    fair_probs = torch.ones([6]) / 6  # 概率向量
    print(multinomial.Multinomial(1, fair_probs).sample())  # 1 是 total count的参数
    counts = multinomial.Multinomial(1000, fair_probs).sample()  # 1000 是 total count的参数
    print(counts / 1000)  # 查看真实事件的概率

    # 到这些概率如何随着时间的推移收敛到真实概率。
    counts = multinomial.Multinomial(10, fair_probs).sample((1000,))  # 让我们进行500组实验，每组抽取10个样本。
    # print(counts)
    cum_counts = counts.cumsum(dim=0)  # 用0维度累加 若0维度上是0，1，2， cumsum 就是 0,1,3
    estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True) # estimates 在0维度上体现了随group数的增大，概率的收敛
    # print(estimates.size()) # 500,6

    # 画图
    for i in range(6):
        plt.plot(estimates[:, i].numpy(),label=("P(die=" + str(i + 1) + ")"))  # 把 每一个概率画成单独的线
        plt.axhline(y=0.167, color='black', linestyle='dashed')
    plt.xlabel('Groups of experiments')
    plt.ylabel('Estimated probability')
    plt.legend();
    plt.show()



if __name__ == '__main__':
    test1()
