import itertools

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import (CosineAnnealingLR,
                                      CosineAnnealingWarmRestarts, LambdaLR)

initial_lr = 0.1
total_epoch = 100


class StepLR:
    def __init__(self, optimizer, learning_rate: float, total_epochs: int):
        self.optimizer = optimizer
        self.total_epochs = total_epochs
        self.base = learning_rate

    def __call__(self, epoch):
        if epoch < self.total_epochs * 3/10:
            lr = self.base
        elif epoch < self.total_epochs * 6/10:
            lr = self.base * 0.2
        elif epoch < self.total_epochs * 8/10:
            lr = self.base * 0.2 ** 2
        else:
            lr = self.base * 0.2 ** 3

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]


class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3)

    def forward(self, x):
        pass


net = model()

optimizer = torch.optim.Adam(net.parameters(), lr=initial_lr)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
# scheduler = StepLR(optimizer, initial_lr, total_epoch)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5)
# scheduler = LambdaLR(optimizer, lambda step : (1.0-step/total_epoch), last_epoch=-1)
# scheduler = CosineAnnealingWarmRestarts(optimizer,T_0=5,T_mult=2)
# scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
#                                               lambda step: (1.0-step/total_epoch) if step <= total_epoch else 0, last_epoch=-1)
print("初始化的学习率：", optimizer.defaults['lr'])

lr_list = []  # 把使用过的lr都保存下来，之后画出它的变化

for epoch in range(1, total_epoch):
    optimizer.zero_grad()
    optimizer.step()
    print("第%d个epoch的学习率：%f" % (epoch, optimizer.param_groups[0]['lr']))
    print(scheduler.get_lr())
    lr_list.append(optimizer.param_groups[0]['lr'])
    # lr_list.append(scheduler.get_last_lr()[0])
    scheduler.step()
    # scheduler(epoch)

# 画出lr的变化
plt.plot(list(range(1, total_epoch)), lr_list)
plt.xlabel("epoch")
plt.ylabel("lr")
plt.title("learning rate's curve changes as epoch goes on!")
plt.show()
