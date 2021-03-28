import random

from paddle.vision.transforms import (BrightnessTransform, ContrastTransform,
                                      HueTransform, RandomHorizontalFlip,
                                      RandomResizedCrop, SaturationTransform,
                                      ToTensor)
from paddle_resnet20 import ResNet20
import paddle 

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

resnet20 = ResNet20(100, channel_list)

model = paddle.Model(resnet20)

model.prepare(paddle.optimizer.Adam(parameters=model.parameters()),
              paddle.nn.CrossEntropyLoss(),
              paddle.metric.Accuracy())

data_file = './data/cifar-100-python.tar.gz'

transforms = paddle.vision.transforms.Compose([
    RandomHorizontalFlip(),
    RandomResizedCrop((32, 32)),
    SaturationTransform(0.2),
    BrightnessTransform(0.2),
    ContrastTransform(0.2),
    HueTransform(0.2), 
    ToTensor()
])

train_dataset = paddle.vision.datasets.Cifar100(
    data_file, mode='train', transform=transforms)
test_dataset = paddle.vision.datasets.Cifar100(
    data_file, mode='test', transform=ToTensor())

model.fit(train_dataset, test_dataset, epochs=100, batch_size=4, verbose=1)
