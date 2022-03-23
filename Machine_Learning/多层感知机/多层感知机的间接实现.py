import torch
from torch import nn
from d2l import torch as d2l
from tools.trainer import basic_trainer, show_images
import os

os.environ["OMP_NUM_THREADS"] = "1"


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


if __name__ == '__main__':

    net = nn.Sequential(nn.Flatten(),
                        nn.Linear(784, 256),
                        nn.ReLU(),
                        nn.Linear(256, 10))

    net.apply(init_weights)
    batch_size, lr, num_epochs = 256, 0.1, 10
    loss = nn.CrossEntropyLoss(reduction='none')
    updater = torch.optim.SGD(net.parameters(), lr=lr)

    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    basic_trainer(net, train_iter, test_iter, loss, num_epochs, updater)