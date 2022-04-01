# 虽然AlexNet证明深层神经网络卓有成效，但它没有提供一个通用的模板来指导后续的研究人员设计新的网络。
import torch
from torch import nn
from d2l import torch as d2l
from tools import trainer

"""
虽然AlexNet证明深层神经网络卓有成效，但它没有提供一个通用的模板来指导后续的研究人员设计新的网络。

经典卷积神经网络的基本组成部分是下面的这个序列：

带填充以保持分辨率的卷积层；
非线性激活函数，如ReLU；
汇聚层，如最大汇聚层。

VGG块与之类似，由一系列卷积层组成
"""


# 定义一个VGG块
# n 个卷积层加一个池化层
def vgg_block(num_convs, in_channels, out_channels):  # 卷积层的数量num_convs、输入通道的数量in_channels 输出通道的数量out_channels.
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels,
                                kernel_size=3, padding=1))  # n -3 +2+1 ，经过卷积大小不变
        layers.append(nn.ReLU())
        in_channels = out_channels  # 修改通道数
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))  # size/2
    return nn.Sequential(*layers)


# 超参数变量
# VGG-11 8个卷积层，3个全连接层
conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))  # (块中卷积层数，通道数) 两个块各有一个卷积层，后三个块各包含两个卷积层


# VGG-11
def vgg(conv_arch):
    conv_blks = []
    in_channels = 1
    # 卷积层部分
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels  # 互换

    return nn.Sequential(
        *conv_blks, nn.Flatten(),
        # 全连接层部分
        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 10))


net = vgg(conv_arch)
X = torch.randn(size=(1, 1, 224, 224))
for blk in net:
    X = blk(X)
    print(blk.__class__.__name__, 'output shape:\t', X.shape)

# 构建一个通道数更小的vgg网络
ratio = 4
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
net = vgg(small_conv_arch)
lr, num_epochs, batch_size = 0.05, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
trainer.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())