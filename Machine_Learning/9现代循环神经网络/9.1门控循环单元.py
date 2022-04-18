"""
梯度异常在实践中的意义
    1. 早期观测值对预测未来所有观测值具有非常重要的意义，
        会导致早期的观测值有着非常大的梯度。
        我们需要一个记忆元去存储重要的早期信息
    2. 一些词元没有观测值
        需要一个机制跳过隐藏状态中此类词元
    3. 逻辑中断。比如新的章节、熊市和牛市

门控隐单元
    这意味着模型有专门的机制来确定应该何时更新隐状态， 以及应该何时重置隐状态。
    这个机制是可学习的

重置门和更新门
    - 是 (0,1)区间中的向量
    - 重置门控制"可能还想记住"的数量
    - 更新门控制"新状态中有多少个旧状态"
     两个门的输出是由使用sigmoid激活函数的两个全连接层给出。
     Rt = sigmoid(Xt * Wxr + Ht-1 * Whr + br) # 是和隐藏状态一样的向量

重置门有助于捕获序列中的短期依赖关系。
更新门有助于捕获序列中的长期依赖关系。

候选隐藏状态(重置门，控制我前面的信息忘记了多少)
    - H = tanh(Wt * Wxh + (R Hadamard Ht-1)Whh + bn)
    当趋向于1时，前面的信息全部都要，趋向于RNN情况
    当趋向于0时，前面的信息全部放弃，重置隐状态

隐状态(更新门，控制我这次更新的权重有多少)
     - Ht = Zt Hadamard Ht-1 + (1-Zt) Hadamard H
     当趋向于1时，是属于遗忘更新 Ht = Ht-1
     当趋向于0时，趋向于RNN


"""

# 从零开始实现
import torch
from torch import nn
from d2l import torch as d2l

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

# 初始化模型参数
def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device)*0.01

    def three():
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                torch.zeros(num_hiddens, device=device))

    W_xz, W_hz, b_z = three()  # 更新门参数
    W_xr, W_hr, b_r = three()  # 重置门参数
    W_xh, W_hh, b_h = three()  # 候选隐状态参数
    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    # 附加梯度
    params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params

def init_gru_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), )

def gru(inputs, state, params):
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        Z = torch.sigmoid((X @ W_xz) + (H @ W_hz) + b_z)
        R = torch.sigmoid((X @ W_xr) + (H @ W_hr) + b_r)
        H_tilda = torch.tanh((X @ W_xh) + ((R * H) @ W_hh) + b_h)
        H = Z * H + (1 - Z) * H_tilda
        Y = H @ W_hq + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)

vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
num_epochs, lr = 500, 1
model = d2l.RNNModelScratch(len(vocab), num_hiddens, device, get_params,
                            init_gru_state, gru)
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)

# 简介实现
num_inputs = vocab_size
gru_layer = nn.GRU(num_inputs, num_hiddens)
model = d2l.RNNModel(gru_layer, len(vocab))
model = model.to(device)
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)