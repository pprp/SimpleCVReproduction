import random

from paddle.vision.transforms import (BrightnessTransform, ContrastTransform,
                                      HueTransform, RandomHorizontalFlip,
                                      RandomResizedCrop, SaturationTransform,
                                      ToTensor)
from paddle_resnet20 import ResNet20
import paddle

import json

json_result = {}

f = open("log/save_arc_acc_%d.json" % (random.randint(0,10000)), "w")

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
    DATA_FILE = './data/data76994/cifar-100-python.tar.gz'

    model.prepare(
        paddle.optimizer.Momentum(
            learning_rate=LinearWarmup(CosineAnnealingDecay(LR, MAX_EPOCH), 2000, 0., LR),
            momentum=MOMENTUM,
            parameters=model.parameters(),
            weight_decay=WEIGHT_DECAY),
        paddle.nn.CrossEntropyLoss(),
        paddle.metric.Accuracy(topk=(1,5)))

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
    callbacks = [LRSchedulerM(), callbacks.VisualDL('vis_logs/res20_3x3_lr0.1cos_e300_bs128_bri_con_aug')]
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
    result = model.evaluate(test_dataset, batch_size=64)
    print(ii, '\t', channel_list, '\t', result['acc'])

    json_result["arch%d" % ii] = {
        "arc": channel_list,
        "acc": result["acc"]
    }

    json.dump(json_result, f)


f.close()
