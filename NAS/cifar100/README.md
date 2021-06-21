# Channel Search for ResNet20

[TOC]

## Feature

- [x] support channel search for resnet20
- [x] support artificial design model such as resnet, densenet, mobilenet,shufflenet,DLA,DPN etc. 
- [ ] support automl models such as regnet, pnasnet
- [ ] support data augmentation tricks:
	- [ ] Cutout
	- [ ] AutoAugmentation
	- [ ] Shake-Shake
	- [ ] Mixup
	- [ ] random erasing
	- [ ] auto-augment
	- [ ] rand-augment
- [ ] experimental results on cifar100 and cifar10
- [ ] distributed data parrallel to support
- [ ] support transformer models
- [ ] support differenet datasets such as tiny imagenet

## Get Started





## Experimental Results

Training Details:

- total epoch: 300 
- lr scheduler: CosineAnnealingLR
- weight decay: 5e-4

### CIFAR10

| Network         | Params(M) | Train loss | Train top1 | Val loss | Val top1 | Hyper               | GPU(M) |
| --------------- | --------- | ---------- | ---------- | -------- | -------- | ------------------- | ------ |
| densenet_cifar  | 4.4       | 0.00156    | 99.99%     | 0.24     | 94.83%   | 0.1/256/w/o cutout  | 7303   |
| dla             | 63        | 0.00164    | 99.99%     | 0.20     | 95.57%   | 0.1/256/w/o cutout  | 5555   |
| resnet50        | 91        | 0.00105    | 100.0%     | 0.19     | 95.74%   | 0.1/256/w/o cutout  | 10895  |
| attention52     | 214       | 0.00109    | 99.99%     | 0.49     | 90.62%   | 0.01/256/w/o cutout | 5691   |
| dpn26           | 45        | 0.00195    | 100.0%     | 0.16     | 95.43%   | 0.1/256/w/o cutout  | 10260  |
| resnet50_cutout | 91        |            |            |          |          | 0.1/128/ cutout=0.5 | 10895  |
| efficientnetb0  |           | 0.02396    | 99.32%     | 0.35     | 91.52%   | 0.1/128/w/o cutout  | 3961   |
|                 |           |            |            |          |          |                     |        |
|                 |           |            |            |          |          |                     |        |





### CIFAR100

| dataset  | network            | params | top1 err | top5 err | epoch(lr = 0.1) | epoch(lr = 0.02) | epoch(lr = 0.004) | epoch(lr = 0.0008) | total epoch |
| -------- | ------------------ | ------ | -------- | -------- | --------------- | ---------------- | ----------------- | ------------------ | ----------- |
| cifar100 | mobilenet          | 3.3M   | 34.02    | 10.56    | 60              | 60               | 40                | 40                 | 200         |
| cifar100 | mobilenetv2        | 2.36M  | 31.92    | 09.02    | 60              | 60               | 40                | 40                 | 200         |
| cifar100 | squeezenet         | 0.78M  | 30.59    | 8.36     | 60              | 60               | 40                | 40                 | 200         |
| cifar100 | shufflenetv2       | 1.3M   | 30.49    | 8.49     | 60              | 60               | 40                | 40                 | 200         |
| cifar100 | vgg13_bn           | 28.7M  | 28.00    | 9.71     | 60              | 60               | 40                | 40                 | 200         |
| cifar100 | vgg16_bn           | 34.0M  | 27.07    | 8.84     | 60              | 60               | 40                | 40                 | 200         |
| cifar100 | resnet18           | 11.2M  | 24.39    | 6.95     | 60              | 60               | 40                | 40                 | 200         |
| cifar100 | resnet34           | 21.3M  | 23.24    | 6.63     | 60              | 60               | 40                | 40                 | 200         |
| cifar100 | resnet50           | 23.7M  | 22.61    | 6.04     | 60              | 60               | 40                | 40                 | 200         |
| cifar100 | resnet101          | 42.7M  | 22.22    | 5.61     | 60              | 60               | 40                | 40                 | 200         |
| cifar100 | resnet152          | 58.3M  | 22.31    | 5.81     | 60              | 60               | 40                | 40                 | 200         |
| cifar100 | preactresnet18     | 11.3M  | 27.08    | 8.53     | 60              | 60               | 40                | 40                 | 200         |
| cifar100 | preactresnet34     | 21.5M  | 24.79    | 7.68     | 60              | 60               | 40                | 40                 | 200         |
| cifar100 | preactresnet50     | 23.9M  | 25.73    | 8.15     | 60              | 60               | 40                | 40                 | 200         |
| cifar100 | preactresnet101    | 42.9M  | 24.84    | 7.83     | 60              | 60               | 40                | 40                 | 200         |
| cifar100 | preactresnet152    | 58.6M  | 22.71    | 6.62     | 60              | 60               | 40                | 40                 | 200         |
| cifar100 | resnext50          | 14.8M  | 22.23    | 6.00     | 60              | 60               | 40                | 40                 | 200         |
| cifar100 | resnext101         | 25.3M  | 22.22    | 5.99     | 60              | 60               | 40                | 40                 | 200         |
| cifar100 | resnext152         | 33.3M  | 22.40    | 5.58     | 60              | 60               | 40                | 40                 | 200         |
| cifar100 | attention59        | 55.7M  | 33.75    | 12.90    | 60              | 60               | 40                | 40                 | 200         |
| cifar100 | attention92        | 102.5M | 36.52    | 11.47    | 60              | 60               | 40                | 40                 | 200         |
| cifar100 | densenet121        | 7.0M   | 22.99    | 6.45     | 60              | 60               | 40                | 40                 | 200         |
| cifar100 | densenet161        | 26M    | 21.56    | 6.04     | 60              | 60               | 60                | 40                 | 200         |
| cifar100 | densenet201        | 18M    | 21.46    | 5.9      | 60              | 60               | 40                | 40                 | 200         |
| cifar100 | googlenet          | 6.2M   | 21.97    | 5.94     | 60              | 60               | 40                | 40                 | 200         |
| cifar100 | inceptionv3        | 22.3M  | 22.81    | 6.39     | 60              | 60               | 40                | 40                 | 200         |
| cifar100 | inceptionv4        | 41.3M  | 24.14    | 6.90     | 60              | 60               | 40                | 40                 | 200         |
| cifar100 | inceptionresnetv2  | 65.4M  | 27.51    | 9.11     | 60              | 60               | 40                | 40                 | 200         |
| cifar100 | xception           | 21.0M  | 25.07    | 7.32     | 60              | 60               | 40                | 40                 | 200         |
| cifar100 | seresnet18         | 11.4M  | 23.56    | 6.68     | 60              | 60               | 40                | 40                 | 200         |
| cifar100 | seresnet34         | 21.6M  | 22.07    | 6.12     | 60              | 60               | 40                | 40                 | 200         |
| cifar100 | seresnet50         | 26.5M  | 21.42    | 5.58     | 60              | 60               | 40                | 40                 | 200         |
| cifar100 | seresnet101        | 47.7M  | 20.98    | 5.41     | 60              | 60               | 40                | 40                 | 200         |
| cifar100 | seresnet152        | 66.2M  | 20.66    | 5.19     | 60              | 60               | 40                | 40                 | 200         |
| cifar100 | nasnet             | 5.2M   | 22.71    | 5.91     | 60              | 60               | 40                | 40                 | 200         |
| cifar100 | stochasticdepth18  | 11.22M | 31.40    | 8.84     | 60              | 60               | 40                | 40                 | 200         |
| cifar100 | stochasticdepth34  | 21.36M | 27.72    | 7.32     | 60              | 60               | 40                | 40                 | 200         |
| cifar100 | stochasticdepth50  | 23.71M | 23.35    | 5.76     | 60              | 60               | 40                | 40                 | 200         |
| cifar100 | stochasticdepth101 | 42.69M | 21.28    | 5.39     | 60              | 60               | 40                | 40                 | 200         |






## Thanks

https://github.com/KaiyangZhou/deep-person-reid

https://github.com/weiaicunzai/pytorch-cifar100

https://github.com/kuangliu/pytorch-cifar

