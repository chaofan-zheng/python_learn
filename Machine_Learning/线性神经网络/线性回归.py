import math
import time
import numpy as np
import torch


class Timer:  # 自定义计时器
    """记录多次运行时间"""

    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """启动计时器"""
        self.tik = time.time()

    def stop(self):
        """停止计时器并将时间记录在列表中"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和"""
        return sum(self.times)

    def cumsum(self):
        """返回累计时间"""
        return np.array(self.times).cumsum().tolist()


def test1():
    """
    矢量化加速，避开代价高昂的for循环
    :return:
    """
    n = 10000
    a = np.ones(n)
    b = np.ones(n)

    # 使用for循环相加
    c = np.zeros(n)
    timer = Timer()
    for i in range(n):
        c[i] = a[i] + b[i]
    print(f'{timer.stop():.5f} sec')

    # 使用重载的+来相加
    timer.start()
    d = a + b
    print(f'{timer.stop():.5f} sec')
    # 矢量化代码通常会带来数量级的加速。 另外，我们将更多的数学运算放到库中，而无须自己编写那么多的计算，从而减少了出错的可能性。

def test2():
    """
    :return:
    """
    pass



if __name__ == '__main__':
    test1()
