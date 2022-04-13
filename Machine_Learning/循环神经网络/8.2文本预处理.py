"""
1. 将文本作为字符串加载到内存中。

2. 将字符串拆分为词元（如单词和字符）。

3. 建立一个词表，将拆分的词元映射到数字索引。

4. 将文本转换为数字索引序列，方便模型操作。

"""
import collections
import re
from d2l import torch as d2l

d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt', '090b5e7e70c295757f55df93cb0a180b9691891a')


def read_time_machine():  # @save
    """将时间机器数据集加载到文本行的列表中"""
    with open(d2l.download('time_machine'), 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]


if __name__ == '__main__':
    lines = read_time_machine()
    print(f'# 文本总行数: {len(lines)}')
    print(lines[0])
    print(lines[10])
