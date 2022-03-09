# https://docs.python.org/3/library/argparse.html

import argparse


def example():
    parser = argparse.ArgumentParser(description="处理一些整数")  # 创建一个解析器对象
    parser.add_argument('integers',  # 名称
                        metavar='N',  # 显示在外面的参数名称
                        type=int,
                        nargs='+',  # 应该使用的参数的数量
                        help='an integer for the accumulator')
    parser.add_argument('--sum',
                        dest='accumulate',  # 返回的对象的属性的名称
                        action='store_const', # 在命令行遇到此参数时要采取的基本操作类型。
                        const=sum,
                        default=max,
                        help='sum the integers (default: find the max)')

    args = parser.parse_args()
    print(args)
    print(args.integers)  # [1, 2, 3, 4]
    print(args.accumulate)  # <built-in function max>
    print(args.accumulate(args.integers))
    # python prog.py 1 2 3 4   ----> 4
    # python prog.py 1 2 3 4 --sum  ------> 10


if __name__ == '__main__':
    example()
