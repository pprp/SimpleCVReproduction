# Benchmarks and Checkpoints

Each zip file contains 4 types of files

* a checkpoint of the model, typically, named as `model_best.pth.tar`
* the md5 of the checkpoint
* a hyper-parameter json file, typically, named as `hparams_train.json`
* `tensorboard` log file, you can use `tensorboard` to visualize the log. It is in the `val` directory within the zip file.

Hope you have fun with these checkpoints.
Any issues about checkpoints should be raised at
[![checkpoints](https://img.shields.io/badge/issue-3-yellow)](https://github.com/mzhaoshuai/SplitNet-Divide-and-Co-training/issues/3).


## ImageNet

General training protocols: batch size 256, epochs 120, cos learning rate 0.1, AutoAugment/RandAugment, Label smoothing,
mixup, random erasing.


| Methods                   		| Top-1/Top-5 Acc 		| MParams/GFLOPs      | Checkpoints  |
|-----------------------------------|-----------------------|---------------------|--------------|
| ResNet-50, 224px           		| 78.84 / 94.47         | 25.7 / 5.5       	  | [resnet50_split1_imagenet_256_06](https://github.com/mzhaoshuai/SplitNet-Divide-and-Co-training/releases/download/1.0.0/resnet50_split1_imagenet_256_06.zip) |
| SE-ResNet-50, 224px           	| 79.47 / 94.54         | 28.2 / 4.9      	  | [se_resnet50_split1_imagenet_256_01](https://github.com/mzhaoshuai/SplitNet-Divide-and-Co-training/releases/download/1.0.0/se_resnet50_split1_imagenet_256_01.zip) |
| ResNeXSt-50, 4x16d, 224px         | 79.85 / 94.98         | 17.8 / 4.3      	  | [resnexst50_4x16d_split1_imagenet_256_01](https://github.com/mzhaoshuai/SplitNet-Divide-and-Co-training/releases/download/1.0.0/resnexst50_4x16d_split1_imagenet_256_01.zip) |
| ResNeXSt-50, 8x16d, 224px         | 80.90 / 95.36         | 30.5 / 6.8      	  | [resnexst50_8x16d_split1_imagenet_256_03](https://github.com/mzhaoshuai/SplitNet-Divide-and-Co-training/releases/download/1.0.0/resnexst50_8x16d_split1_imagenet_256_03.zip) |
| ResNeXSt-50, 4x32d, 224px         | 81.10 / 95.49         | 37.1 / 8.3      	  | [resnexst50_4x32d_split1_imagenet_256_05](https://github.com/mzhaoshuai/SplitNet-Divide-and-Co-training/releases/download/1.0.0/resnexst50_4x32d_split1_imagenet_256_05.zip) |
| ResNet-110, 224px      		 	| 80.16 / 94.54         | 44.8 / 9.2          | [resnet101_split1_imagenet_256_01](https://github.com/mzhaoshuai/SplitNet-Divide-and-Co-training/releases/download/1.0.0/resnet101_split1_imagenet_256_01.zip) |
| WRN-50-2, 224px           		| 80.66 / 95.16         | 68.9 / 12.8         | [wide_resnet50_2_split1_imagenet_256_01](https://github.com/mzhaoshuai/SplitNet-Divide-and-Co-training/releases/download/1.0.0/wide_resnet50_2_split1_imagenet_256_01.zip) |
| WRN-50-2, S=2, 224px      		| 79.64 / 94.82         | 51.4 / 10.9         | [wide_resnet50_2_split2_imagenet_256_02](https://github.com/mzhaoshuai/SplitNet-Divide-and-Co-training/releases/download/1.0.0/wide_resnet50_2_split2_imagenet_256_02.zip) | 
| WRN-50-3, 224px           		| 80.74 / 95.40         | 135.0 / 23.8        | [wide_resnet50_3_split1_imagenet_256_01](https://github.com/mzhaoshuai/SplitNet-Divide-and-Co-training/releases/download/1.0.0/wide_resnet50_3_split1_imagenet_256_01.zip) |
| WRN-50-3, S=2, 224px      		| 81.42 / 95.62         | 138.0 / 25.6        | [wide_resnet50_3_split2_imagenet_256_02](https://github.com/mzhaoshuai/SplitNet-Divide-and-Co-training/releases/download/1.0.0/wide_resnet50_3_split2_imagenet_256_02.zip) | 
| ResNeXt-101, 64x4d, 224px 		| 81.57 / 95.73         | 83.6 / 16.9         | [resnext101_64x4d_split1_imagenet_256_01](https://github.com/mzhaoshuai/SplitNet-Divide-and-Co-training/releases/download/1.0.0/resnext101_64x4d_split1_imagenet_256_01.zip) |
| ResNeXt-101, 64x4d, S=2, 224px  	| 82.13 / 95.98         | 88.6 / 18.8         | [resnext101_64x4d_split2_imagenet_256_02](https://github.com/mzhaoshuai/SplitNet-Divide-and-Co-training/releases/download/1.0.0/resnext101_64x4d_split2_imagenet_256_02.zip) |
| EfficientNet-B7, 320px 			| 81.83 / 95.78         | 66.7 / 10.6         | [efficientnetb7_split1_imagenet_128_03](https://github.com/mzhaoshuai/SplitNet-Divide-and-Co-training/releases/download/1.0.0/efficientnetb7_split1_imagenet_128_03.zip) |
| EfficientNet-B7, S=2, 320px  		| 82.74 / 96.30         | 68.2 / 10.5         | [efficientnetb7_split2_imagenet_128_02](https://github.com/mzhaoshuai/SplitNet-Divide-and-Co-training/releases/download/1.0.0/efficientnetb7_split2_imagenet_128_02.zip) | 
| SE-ResNeXt-101, 64x4d, S=2, 416px, 120 epochs | 83.34 / 96.61         | 98.0 / 61.1         | [se_resnext101_64x4d_split2_imagenet_128_02](https://github.com/mzhaoshuai/SplitNet-Divide-and-Co-training/releases/download/1.0.0/se_resnext101_64x4d_split2_imagenet_128_02.zip) |
| SE-ResNeXt-101, 64x4d, S=2, 320px, 350 epochs | 83.60 / 96.69         | 98.0 / 38.2         | [se_resnext101_64x4d_B_split2_imagenet_128_05](https://github.com/mzhaoshuai/SplitNet-Divide-and-Co-training/releases/download/1.0.0/se_resnext101_64x4d_B_split2_imagenet_128_05.zip) |

## CIFAR-100

| Methods                   | Top-1 Acc 	| MParams/GFLOPs    | Checkpoints  |
|---------------------------|----------------|---------------------|--------------|  
| WRN-28-10            		| 84.50          | 36.5 / 5.25         | [wide_resnet28_10_split1_cifar100_128_01_acc84.5](https://github.com/mzhaoshuai/SplitNet-Divide-and-Co-training/releases/download/1.0.1/wide_resnet28_10_split1_cifar100_128_01_acc84.5.zip) |
| WRN-28-10, S=2            | 85.52          | 35.8 / 5.16         | [wide_resnet28_10_split2_cifar100_128_02_acc85.52](https://github.com/mzhaoshuai/SplitNet-Divide-and-Co-training/releases/download/1.0.1/wide_resnet28_10_split2_cifar100_128_02_acc85.52.zip) |
| WRN-28-10, S=4            | 85.74          | 36.7 / 5.28         | [wide_resnet28_10_split4_cifar100_128_03_acc85.74](https://github.com/mzhaoshuai/SplitNet-Divide-and-Co-training/releases/download/1.0.1/wide_resnet28_10_split4_cifar100_128_03_acc85.74.zip) |
| WRN-40-10            		| 83.98          | 55.9 / 8.08         | [wide_resnet40_10_split1_cifar100_128_06_acc83.98](https://github.com/mzhaoshuai/SplitNet-Divide-and-Co-training/releases/download/1.0.1/wide_resnet40_10_split1_cifar100_128_06_acc83.98.zip) |
| WRN-40-10, S=2            | 85.91          | 54.8 / 7.94         | [wide_resnet40_10_split2_cifar100_128_05_acc85.91](https://github.com/mzhaoshuai/SplitNet-Divide-and-Co-training/releases/download/1.0.1/wide_resnet40_10_split2_cifar100_128_05_acc85.91.zip) |
| WRN-40-10, S=4            | 86.90          | 56.0 / 8.12         | [wide_resnet40_10_split4_cifar100_128_04_acc86.90](https://github.com/mzhaoshuai/SplitNet-Divide-and-Co-training/releases/download/1.0.1/wide_resnet40_10_split4_cifar100_128_04_acc86.90.zip) |
| DenseNet-BC-190      		| 85.90          | 25.8 / 9.39         | [densenet190_split1_cifar100_64_01_acc85.90](https://github.com/mzhaoshuai/SplitNet-Divide-and-Co-training/releases/download/1.0.1/densenet190_split1_cifar100_64_01_acc85.90.zip) |
| DenseNet-BC-190, S=2      | 87.36          | 25.5 / 9.24         | [densenet190_split2_cifar100_128_02_acc87.36](https://github.com/mzhaoshuai/SplitNet-Divide-and-Co-training/releases/download/1.0.1/densenet190_split2_cifar100_128_02_acc87.36.zip) |
| DenseNet-BC-190, S=4      | 87.44          | 26.3 / 9.48         | [densenet190_split4_cifar100_64_03_acc87.44](https://github.com/mzhaoshuai/SplitNet-Divide-and-Co-training/releases/download/1.0.1/densenet190_split4_cifar100_64_03_acc87.44.zip) |
| PyramidNet-272       		| 88.98          | 26.8 / 4.55         | [pyramidnet272_split1_cifar100_128_01_88.98](https://github.com/mzhaoshuai/SplitNet-Divide-and-Co-training/releases/download/1.0.1/pyramidnet272_split1_cifar100_128_01_88.98.zip) |
| PyramidNet-272, S=2       | 89.25          | 28.9 / 5.24         | [pyramidnet272_split2_cifar100_128_06_acc89.25](https://github.com/mzhaoshuai/SplitNet-Divide-and-Co-training/releases/download/1.0.1/pyramidnet272_split2_cifar100_128_06_acc89.25.zip) |
| PyramidNet-272, S=4       | 89.46          | 32.8 / 6.33         | [pyramidnet272_split4_cifar100_128_07_acc89.46](https://github.com/mzhaoshuai/SplitNet-Divide-and-Co-training/releases/download/1.0.1/pyramidnet272_split4_cifar100_128_07_acc89.46.zip) |


## CIFAR-10

| Methods                   | Top-1 Acc | MParams/GFLOPs    | Checkpoints  |
|---------------------------|----------------|---------------------|--------------|
| WRN-28-10 				| 97.59          | 36.5 / 5.25         | [wide_resnet28_10_split1_cifar10_128_08_acc97.59](https://github.com/mzhaoshuai/SplitNet-Divide-and-Co-training/releases/download/1.0.2/wide_resnet28_10_split1_cifar10_128_08_acc97.59.zip) |
| WRN-28-10, S=2            | 98.19          | 35.8 / 5.16         | [wide_resnet28_10_split2_cifar10_128_07_acc98.19](https://github.com/mzhaoshuai/SplitNet-Divide-and-Co-training/releases/download/1.0.2/wide_resnet28_10_split2_cifar10_128_07_acc98.19.zip) |
| WRN-28-10, S=4            | 98.32          | 36.5 / 5.28         | [wide_resnet28_10_split4_cifar10_128_24_acc98.32](https://github.com/mzhaoshuai/SplitNet-Divide-and-Co-training/releases/download/1.0.2/wide_resnet28_10_split4_cifar10_128_24_acc98.32.zip) |
| WRN-40-10 				| 97.81          | 55.8 / 8.08         | [wide_resnet40_10_split1_cifar10_128_04_acc97.81](https://github.com/mzhaoshuai/SplitNet-Divide-and-Co-training/releases/download/1.0.2/wide_resnet40_10_split1_cifar10_128_04_acc97.81.zip) |
| WRN-40-10, S=4            | 98.38          | 55.9 / 8.12         | [wide_resnet40_10_split4_cifar10_128_05_acc98.38](https://github.com/mzhaoshuai/SplitNet-Divide-and-Co-training/releases/download/1.0.2/wide_resnet40_10_split4_cifar10_128_05_acc98.38.zip) |
| Shake-Shake 26 2x96d	 	| 98.00          | 26.2 / 3.78         | [shake_resnet26_2x96d_split1_cifar10_128_07_acc98.00](https://github.com/mzhaoshuai/SplitNet-Divide-and-Co-training/releases/download/1.0.2/shake_resnet26_2x96d_split1_cifar10_128_07_acc98.00.zip) |
| Shake-Shake 26 2x96d, S=2 | 98.25          | 23.3 / 3.38         | [shake_resnet26_2x96d_split2_cifar10_128_12](https://github.com/mzhaoshuai/SplitNet-Divide-and-Co-training/releases/download/1.0.2/shake_resnet26_2x96d_split2_cifar10_128_12_acc98.25.zip) |
| Shake-Shake 26 2x96d, S=4 | 98.31          | 26.3 / 3.81         | [shake_resnet26_2x96d_split4_cifar10_128_09](https://github.com/mzhaoshuai/SplitNet-Divide-and-Co-training/releases/download/1.0.2/shake_resnet26_2x96d_split4_cifar10_128_09_acc98.31.zip) |
| PyramidNet-272            | 98.67          | 26.2 / 4.55         | [pyramidnet272_split1_cifar10_128_01_acc98.67](https://github.com/mzhaoshuai/SplitNet-Divide-and-Co-training/releases/download/1.0.2/pyramidnet272_split1_cifar10_128_01_acc98.67.zip) |
| PyramidNet-272, S=4       | 98.71          | 32.6 / 6.33         | [pyramidnet272_split4_cifar10_128_05_acc98.71](https://github.com/mzhaoshuai/SplitNet-Divide-and-Co-training/releases/download/1.0.2/pyramidnet272_split4_cifar10_128_05_acc98.71.zip) |