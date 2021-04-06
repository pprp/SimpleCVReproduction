import json
import math
import random

import numpy as np

import paddle
import paddle.nn as nn
from paddle import callbacks
from paddle.io import DataLoader
from paddle.optimizer.lr import (CosineAnnealingDecay, LinearWarmup,
                                 MultiStepDecay)
from paddle.vision.datasets import Cifar100
from paddle.vision.transforms import (BrightnessTransform, Compose,
                                      ContrastTransform, HueTransform,
                                      Normalize, RandomCrop,
                                      RandomHorizontalFlip, RandomResizedCrop,
                                      RandomRotation, SaturationTransform,
                                      ToTensor)
from paddle_resnet20 import ResNet20


class ToArray(object):
    def __call__(self, img):
        img = np.array(img)
        img = np.transpose(img, [2, 0, 1])
        img = img / 255.
        return img.astype('float32')


class RandomApply(object):
    def __init__(self, transform, p=0.5):
        super().__init__()
        self.p = p
        self.transform = transform

    def __call__(self, img):
        if self.p < random.random():
            return img
        img = self.transform(img)
        return img


class LRSchedulerM(callbacks.LRScheduler):
    def __init__(self, by_step=False, by_epoch=True, warm_up=True):
        super().__init__(by_step, by_epoch)
        assert by_step ^ warm_up
        self.warm_up = warm_up

    def on_epoch_end(self, epoch, logs=None):
        if self.by_epoch and not self.warm_up:
            if self.model._optimizer and hasattr(
                self.model._optimizer, '_learning_rate') and isinstance(
                    self.model._optimizer._learning_rate, paddle.optimizer.lr.LRScheduler):
                self.model._optimizer._learning_rate.step()

    def on_train_batch_end(self, step, logs=None):
        if self.by_step or self.warm_up:
            if self.model._optimizer and hasattr(
                self.model._optimizer, '_learning_rate') and isinstance(
                    self.model._optimizer._learning_rate, paddle.optimizer.lr.LRScheduler):
                self.model._optimizer._learning_rate.step()
            if self.model._optimizer._learning_rate.last_epoch >= self.model._optimizer._learning_rate.warmup_steps:
                self.warm_up = False


def _on_train_batch_end(self, step, logs=None):
    logs = logs or {}
    logs['lr'] = self.model._optimizer.get_lr()
    self.train_step += 1
    if self._is_write():
        self._updates(logs, 'train')


def _on_train_begin(self, logs=None):
    self.epochs = self.params['epochs']
    assert self.epochs
    self.train_metrics = self.params['metrics'] + ['lr']
    assert self.train_metrics
    self._is_fit = True
    self.train_step = 0


callbacks.VisualDL.on_train_batch_end = _on_train_batch_end
callbacks.VisualDL.on_train_begin = _on_train_begin
json_result = {}

f = open("log/save_arc_acc_%d.json" % (random.randint(0, 10000)), "w")

for ii in range(300):

    channel_list = []
    for i in range(1, 21):
        if 0 < i <= 7:
            channel_list.append(random.choice([4, 8, 12, 16]))
        elif 7 < i <= 13:
            channel_list.append(random.choice([4, 8, 12, 16, 20, 24, 28, 32]))
        elif 13 < i <= 19:
            channel_list.append(random.choice(
                [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64]))
        else:
            channel_list.append(random.choice(
                [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64]))

    print(channel_list)

    resnet20 = ResNet20(100, channel_list)
    model = paddle.Model(resnet20)

    MAX_EPOCH = 300
    LR = 0.1
    WEIGHT_DECAY = 5e-4
    MOMENTUM = 0.9
    BATCH_SIZE = 128
    CIFAR_MEAN = [0.5071, 0.4865, 0.4409]
    CIFAR_STD = [0.1942, 0.1918, 0.1958]
    DATA_FILE = './data/cifar-100-python.tar.gz'

    model.prepare(
        paddle.optimizer.Momentum(
            learning_rate=LinearWarmup(
                CosineAnnealingDecay(LR, MAX_EPOCH), 2000, 0., LR),
            momentum=MOMENTUM,
            parameters=model.parameters(),
            weight_decay=WEIGHT_DECAY),
        paddle.nn.CrossEntropyLoss(),
        paddle.metric.Accuracy(topk=(1, 5)))

    transforms = Compose([
        RandomCrop(32, padding=4),
        RandomApply(BrightnessTransform(0.1)),
        RandomApply(ContrastTransform(0.1)),
        RandomHorizontalFlip(),
        RandomRotation(15),
        ToArray(),
        Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    val_transforms = Compose([ToArray(), Normalize(CIFAR_MEAN, CIFAR_STD)])
    train_set = Cifar100(DATA_FILE, mode='train', transform=transforms)
    test_set = Cifar100(DATA_FILE, mode='test', transform=val_transforms)
    callbacks = [LRSchedulerM(), callbacks.VisualDL(
        'vis_logs/res20_3x3_lr0.1cos_e300_bs128_bri_con_aug')]
    model.fit(
        train_set,
        test_set,
        epochs=MAX_EPOCH,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        verbose=1,
        callbacks=callbacks,
    )
    result = model.evaluate(test_set, batch_size=64)
    print(ii, '\t', channel_list, '\t', result['acc'])

    json_result["arch%d" % ii] = {
        "arc": channel_list,
        "acc": result["acc"]
    }

    json.dump(json_result, f)


f.close()
