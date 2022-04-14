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
    with open('../data/timemachine.txt', 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]


def tokenize(lines, token='word'):  # @save
    """将文本行拆分为单词或字符词元"""
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('错误：未知词元类型：' + token)


# 构建词表
"""
我们先将训练集中的所有文档合并在一起，对它们的唯一词元进行统计， 得到的统计结果称之为语料（corpus）。 
然后根据每个唯一词元的出现频率，为其分配一个数字索引。 很少出现的词元通常被移除，这可以降低复杂性。 
另外，语料库中不存在或已删除的任何词元都将映射到一个特定的未知词元“<unk>”
"""


class Vocab:  # @save
    """文本词表"""

    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 按出现频率排序
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)  # 按照出现频率进行排序
        # 未知词元的索引为0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:  # 过滤掉一些出现频率很少的词元，降低复杂性。
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1  # 词元：索引

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property  # 将方法变成只读属性
    def unk(self):  # 未知词元的索引为0
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs


def count_corpus(tokens):  # @save
    """统计词元的频率"""
    # 这里的tokens是1D列表或2D列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 将词元列表展平成一个列表
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)  # 构建成字典，{'token':count}

def load_corpus_time_machine(max_tokens=-1):  #@save
    """
    为了简化后面章节中的训练，我们使用字符（而不是单词）实现文本词元化；
    时光机器数据集中的每个文本行不一定是一个句子或一个段落，还可能是一个单词，因此返回的corpus仅处理为单个列表，而不是使用多词元列表构成的一个列表。
    :param max_tokens:
    :return:
    """
    """返回时光机器数据集的词元索引列表和词表"""
    lines = read_time_machine()
    tokens = tokenize(lines, 'char')  # [[第一行的token],[第二行的token]]
    vocab = Vocab(tokens)
    # 因为时光机器数据集中的每个文本行不一定是一个句子或一个段落，
    # 所以将所有文本行展平到一个列表中
    corpus = [vocab[token] for line in tokens for token in line]  # 获得每一个token的索引
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab

if __name__ == '__main__':
    lines = read_time_machine()
    print(f'# 文本总行数: {len(lines)}')
    print(lines[0])
    print(lines[10])

    # token 是经过拆分的单次或者字符
    tokens = tokenize(lines)
    for i in range(11):
        print(tokens[i])

    # 构建词表，打印前几个高频词元索引（按照频率进行排序）
    vocab = Vocab(tokens)
    print(list(vocab.token_to_idx.items())[:10])

    corpus, vocab = load_corpus_time_machine()
    print(len(corpus), len(vocab))

