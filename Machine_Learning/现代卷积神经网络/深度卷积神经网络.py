import torch
from torch import nn
from d2l import torch as d2l
from tools import trainer
import os

os.environ["OMP_NUM_THREADS"] = "1"


net = nn.Sequential(
    # 这里，我们使用一个11*11的更大窗口来捕捉对象。
    # 同时，步幅为4，以减少输出的高度和宽度。
    # 另外，输出通道的数目远大于LeNet
    nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),  # （224-11+2+4）/4 = 54
    nn.MaxPool2d(kernel_size=3, stride=2),  # 54-3+2/2=26
    # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
    nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),  # 26-5+4+1=26 一致
    nn.MaxPool2d(kernel_size=3, stride=2),  # 26-3+2/2 = 12
    # 使用三个连续的卷积层和较小的卷积窗口。
    # 除了最后的卷积层，输出通道的数量进一步增加。
    # 在前两个卷积层之后，汇聚层不用于减少输入的高度和宽度
    nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),  # 12
    nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),  # 12
    nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),  # 12
    nn.MaxPool2d(kernel_size=3, stride=2),  # 256, 5, 5
    nn.Flatten(),
    # 这里，全连接层的输出数量是LeNet中的好几倍。使用dropout层来减轻过拟合
    nn.Linear(6400, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    # 最后是输出层。由于这里使用Fashion-MNIST，所以用类别数为10，而非论文中的1000
    nn.Linear(4096, 10))

# 查看每一层的输出
X = torch.randn(1, 1, 224, 224)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)


# 训练
batch_size = 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
lr, num_epochs = 0.01, 10
trainer.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())