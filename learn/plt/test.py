from matplotlib import pyplot as plt
import numpy as np


def test():
    x = np.arange(10)
    plt.figure(figsize=(10, 6))
    plt.plot(x, x, label="y = x")
    plt.plot(x, x ** 2, label="y=x^2")
    plt.plot(x, x ** 3, label="y=x^3")
    plt.xlabel("xlabel")
    plt.ylabel("ylabel")
    plt.title("Simple Plot")
    plt.legend()  # 给图加上图例
    plt.show()


if __name__ == '__main__':
    test()
