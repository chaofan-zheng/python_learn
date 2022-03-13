from IPython import display
import numpy as np


def numerical_lim(f, x, h):
    
    return (f(x + h) - f(x)) / h


def f(x):
    return 3 * x ** 2 - 4 * x


def test():
    h = 0.1
    for i in range(5):
        print(f'h={h:.5f}, numerical limit={numerical_lim(f, 1, h):.5f}')
        h *= 0.1


if __name__ == '__main__':
    test()
