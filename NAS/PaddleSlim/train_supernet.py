from paddle.vision.transforms import (
    ToTensor, RandomHorizontalFlip, RandomResizedCrop, SaturationTransform, Compose,
    HueTransform, BrightnessTransform, ContrastTransform, RandomCrop, Normalize, RandomRotation
)
from paddle.vision.datasets import Cifar100
from paddle.io import DataLoader
from paddle.optimizer.lr import CosineAnnealingDecay, MultiStepDecay, LinearWarmup
import random
from resnet20 import *
import paddle

# supernet trainning 基于paddleslim模型压缩包
# https://github.com/PaddlePaddle/PaddleSlim 欢迎大家多多star
from paddleslim.nas.ofa.convert_super import Convert, supernet
from paddleslim.nas.ofa import OFA, RunConfig, DistillConfig
from paddleslim.nas.ofa.utils import utils

channel_list = []
for i in range(1, 21):
    if 0 < i <= 7:
        # channel_list.append(random.choice([ 4, 8, 12, 16]))
        channel_list.append(16)
    elif 7 < i <= 13:
        # channel_list.append(random.choice([ 4, 8, 12, 16, 20, 24, 28, 32]))
        channel_list.append(32)
    elif 13 < i <= 19:
        # channel_list.append(random.choice([ 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56,60, 64]))
        channel_list.append(64)
    else:
        # channel_list.append(random.choice([ 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56,60, 64]))
        channel_list.append(64)

net = ResNet20(100, channel_list)
net2 = ResNet20(100, channel_list)

net2.set_state_dict(paddle.load('./pretrained_model/resnet20.pdparams'))

channel_optional = []
for i in range(0, 23):
    if i <= 7:
        channel_optional.append([4, 8, 12, 16])
        # channel_optional.append([12, 16])
    elif 7 < i <= 14:
        channel_optional.append([4, 8, 12, 16, 20, 24, 28, 32])
        # channel_optional.append([20, 24, 28, 32])
    elif 14 < i <= 21:
        channel_optional.append(
            [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64])
        # channel_optional.append([36, 40, 44, 48, 52, 56,60, 64])
    else:
        channel_optional.append(
            [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64])
        # channel_optional.append([36, 40, 44, 48, 52, 56,60, 64])

distill_config = DistillConfig(teacher_model=net2)
sp_net_config = supernet(channel=channel_optional)
sp_model = Convert(sp_net_config).convert(net)
ofa_net = OFA(sp_model, distill_config=distill_config)
ofa_net.set_task('channel')


model = paddle.Model(ofa_net)

MAX_EPOCH = 300
LR = 0.1
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9
BATCH_SIZE = 128
CIFAR_MEAN = [0.5071, 0.4865, 0.4409]
CIFAR_STD = [0.1942, 0.1918, 0.1958]
DATA_FILE = './data/data76994/cifar-100-python.tar.gz'

model.prepare(
    paddle.optimizer.Momentum(
        learning_rate=LinearWarmup(
            CosineAnnealingDecay(LR, MAX_EPOCH), 2000, 0., LR),
        momentum=MOMENTUM,
        parameters=model.parameters(),
        weight_decay=WEIGHT_DECAY),
    CrossEntropyLoss(),
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
callbacks = [LRSchedulerM(), callbacks.VisualDL('vis_logs/ofa_resnet20')]

model.fit(
    train_set,
    test_set,
    epochs=MAX_EPOCH,
    batch_size=BATCH_SIZE,
    save_dir='checkpoints',
    save_freq=100,
    shuffle=True,
    num_workers=4,
    verbose=1,
    callbacks=callbacks,
)
