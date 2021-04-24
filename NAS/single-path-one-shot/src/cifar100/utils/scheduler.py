import itertools

# import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import (CosineAnnealingLR,
                                      CosineAnnealingWarmRestarts,
                                      ExponentialLR, LambdaLR,
                                      ReduceLROnPlateau, StepLR, _LRScheduler)
from torch.optim.sgd import SGD


class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError(
                'multiplier should be greater thant or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [
                        base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        self.last_epoch = epoch if epoch != 0 else 1
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch /
                                    self.total_epoch + 1.) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)


#     model = [torch.nn.Parameter(torch.randn(2, 2, requires_grad=True))]
#     optim = SGD(model, 0.1)

#     # scheduler_warmup is chained with schduler_steplr
#     scheduler_steplr = StepLR(optim, step_size=10, gamma=0.1)
#     scheduler_warmup = GradualWarmupScheduler(optim, multiplier=1, total_epoch=5, after_scheduler=scheduler_steplr)

#     # this zero gradient update is needed to avoid a warning message, issue #8.
#     optim.zero_grad()
#     optim.step()

#     for epoch in range(1, 20):
#         scheduler_warmup.step(epoch)
#         print(epoch, optim.param_groups[0]['lr'])

#         optim.step()    # backward pass (update network)

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


if __name__ == '__main__':

    initial_lr = 0.1
    total_epoch = 100
    net = model()

    optimizer = torch.optim.Adam(net.parameters(), lr=initial_lr)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    # scheduler = StepLR(optimizer, initial_lr, total_epoch)
    # a_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5)
    a_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                    lambda step: (1.0-step/total_epoch), last_epoch=-1)
    scheduler = GradualWarmupScheduler(
        optimizer, 1, total_epoch=5, after_scheduler=a_scheduler)
    # scheduler = LambdaLR(optimizer, lambda step : (1.0-step/total_epoch), last_epoch=-1)
    # scheduler = CosineAnnealingWarmRestarts(optimizer,T_0=5,T_mult=2)

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
