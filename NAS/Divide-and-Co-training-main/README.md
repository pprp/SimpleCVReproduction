[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![arXiv](https://img.shields.io/badge/cs.CV-%09arXiv%3A2011.14660-red)](https://arxiv.org/abs/2011.14660)

# Divide and Co-training

Divide and co-training achieve 98.71% on CIFAR-10, 89.46% on CIFAR-100, and 83.60% on ImageNet (SE-ResNet-101, 64x4d, 320px)
by dividing one existing large network into several small ones and co-training.

##  Table of Contents

<!--ts-->
* [Introduction](#Introduction)
* [Features and TODO](#Features-and-TODO)
* [Results and Checkpoints](#Results-and-Checkpoints)
    * [Benchmarks and Checkpoints](miscs/checkpoints.md)
* [Installation](#Installation)
* [Training](#Training)
* [Evaluation](#Evaluation)
* [Citations](#Citations)
* [Licenses](#Licenses)
* [Acknowledgements](#Acknowledgements)
<!--te-->


## News
- [2021/01/27] Add new results (83.60%) on ImageNet. Upload a new model, ResNeXSt, a combination of ResNeSt and ResNeXt.


## Introduction

<div align="justify">

This is the code for the paper 
<a href="https://arxiv.org/abs/2011.14660">
Towards Better Accuracy-efficiency Trade-offs: Divide and Co-training.</a>
<br />

The width of a neural network matters since increasing the width
will necessarily increase the model capacity.
However, the performance of a network does not improve linearly
with the width and soon gets saturated.
In this case, we argue that increasing the number of networks (ensemble)
can achieve better accuracy-efficiency trade-offs than purely increasing the width.
To prove it,
one large network is divided into several small ones
regarding its parameters and regularization components.
Each of these small networks has a fraction of the original one's parameters.
We then train these small networks together and make them see various 
views of the same data to increase their diversity.
During this co-training process,
networks can also learn from each other.
As a result, small networks can achieve better ensemble performance
than the large one with few or no extra parameters or FLOPs.
Small networks can also achieve faster inference speed
than the large one by concurrent running on different devices. 
We validate our argument with 8 different neural architectures on
common benchmarks through extensive experiments.
</div>

<div align=center>
  <img src="miscs/fig1_width.png" width="49%"/>
  <img src="miscs/fig3_latency.png" width="49%"/>
</div>


<div align=center>
  <img src="miscs/fig2_framework.png" width="100%"/>
</div>

## Features and TODO

- [x] Support divide and co-training with different models, i.e., ResNet, Wide-ResNet, ResNeXt, ResNeXSt, SENet,
Shake-Shake, DenseNet, PyramidNet (+Shake-Drop), EfficientNet.
- [x] Different data augmentation methods, i.e., mixup, random erasing, auto-augment, rand-augment, cutout
- [x] Distributed training (tested with multi-GPUs on single machine)
- [x] Multi-GPUs synchronized BatchNormalization
- [x] Automated mixed precision training
- [ ] Asynchronous and distributed training of multiple models

We are open to pull requests.


## Results and Checkpoints

### CIFAR-10

<div align=center>
  <img src="miscs/res_cifar10.png" width="50%"/>
</div>

### CIFAR-100

<div align=center>
  <img src="miscs/res_cifar100.png" width="100%"/>
</div>


### ImageNet
Experiments on ImageNet are conducted on a single machine with 8 RTX 2080Ti GPUs or 4 Tesla V100 32GB GPUs.

<div align=center>
   <img src="miscs/res_imagenet.png" width="100%"/>
</div>


### Benchmarks and Checkpoints

[Benchmarks and Checkpoints](miscs/checkpoints.md)

## Installation

* **Install dependencies via docker**

Please install PyTorch-1.6.0 and Python3.6+.
Only PyTorch-1.6.0+ supports built-in AMP training.

We recommend you to use our established PyTorch docker image:
[zhaosssss/torch_lab:1.6.2](https://hub.docker.com/r/zhaosssss/torch_lab).
```
docker pull zhaosssss/torch_lab:1.6.2
```
If you have not installed docker, see https://docs.docker.com/. 


After you install docker and pull our image, you can `cd` to `script` directory and run
```
./run_docker.sh
```
to create a running docker container.

**NOTE**: We map some directories in `run_docker.sh`, if you do not have these directories,
you need to modify the script.
By default, `run_docker.sh` runs container in background
and you need run `docker exec -it ${DOCKER-ID} bash`
to do some interactive operations.

* **Install dependencies via `pip`**

If you do not want to use docker, try
```
pip install -r requirements.txt
```
However, this is not suggested.


### Prepare data

Generally, directories are organized as following:
```
${HOME}
├── dataset             (save the dataset) 
│   │
│   ├── cifar           (dir of CIFAR dataset)
│   ├── imagenet        (dir of ImageNet dataset)
│   └── svhn            (dir of SVHN dataset)
│
├── models              (save the output checkpoints)
│
├── github              (save the code)
│   │   
│   └── splitnet        (the splitnet code repository)
│       │
│       ├── dataset
│       ├── model       
│       ├── script           
│       └── utils 
...
```

- Download [The CIFAR dataset](https://www.cs.toronto.edu/~kriz/cifar.html),
put them in the `dataset/cifar` directory.

- Download [The ImageNet dataset](https://academictorrents.com/details/a306397ccf9c2ead27155983c254227c0fd938e2),
put them in the `dataset/imagenet` directory.

    - Extract the ImageNet dataset following [extract_ILSVRC.sh](https://gist.github.com/BIGBALLON/8a71d225eff18d88e469e6ea9b39cef4).
```
# script to extract ImageNet dataset
# ILSVRC2012_img_train.tar (about 138 GB)
# ILSVRC2012_img_val.tar (about 6.3 GB)
# make sure ILSVRC2012_img_train.tar & ILSVRC2012_img_val.tar in your current directory

# 1. Extract the training data:

mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train
tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
cd ..

# 2. Extract the validation data and move images to subfolders:

mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xvf ILSVRC2012_img_val.tar
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
```

- Download [The SVHN dataset](http://ufldl.stanford.edu/housenumbers/) (*Format 2: Cropped Digits*),
put them in the `dataset/svhn` directory.

- `cd` to `github` directory and clone the `Divide-and-Co-training` repo.
For brevity, rename it as `splitnet`.


## Training

See `script/train_split.sh` for detailed information.
Before start training, you should specify some variables in the `script/train_split.sh`.

For example:

- `arch`, the architecture you want to use.


You can find more information about the arguments of the code in `parser_params.py`.
```
python parser_params.py --help

usage: parser_params.py [-h] [--data DIR] [--model_dir MODEL_DIR]
                        [--arch {resnet34,resnet50,resnet101,resnet110,resnet164,wide_resnet16_8,wide_resnet16_12,wide_resnet28_10,wide_resnet40_10,wide_resnet52_8,wide_resnet50_2,wide_resnet50_3,wide_resnet101_2,resnext50_32x4d,resnext101_32x4d,resnext101_64x4d,resnext29_8x64d,resnext29_16x64d,se_resnet110,se_resnet164,senet154,se_resnet50,se_resnet101,se_resnet152,shake_resnet26_2x96d,pyramidnet272,efficientnetb0,efficientnetb1,efficientnetb2,efficientnetb3,efficientnetb4,efficientnetb5,efficientnetb6,efficientnetb7}]
                        ...
                        [--kl_weight KL_WEIGHT]
                        [--is_kl_weight_warm_up IS_KL_WEIGHT_WARM_UP]
                        [--kl_weight_warm_up_epochs KL_WEIGHT_WARM_UP_EPOCHS]
                        [--is_linear_lr IS_LINEAR_LR]
                        [--is_summary IS_SUMMARY]
                        [--is_train_sep IS_TRAIN_SEP] [--weight_decay W]
                        [--is_wd_test IS_WD_TEST] [--is_div_wd IS_DIV_WD]
                        [--is_wd_all IS_WD_ALL] [--div_wd_den DIV_WD_DEN]
                        [--max_ckpt_nums MAX_CKPT_NUMS]
                        [--is_apex_amp IS_APEX_AMP]
                        [--amp_opt_level AMP_OPT_LEVEL]

PyTorch SplitNet Training

optional arguments:
  -h, --help            show this help message and exit
  --data DIR            path to dataset
  --model_dir MODEL_DIR
                        dir to which model is saved (default: ./model_dir)
  ...
  --max_ckpt_nums MAX_CKPT_NUMS
                        maximum number of ckpts.
  --is_apex_amp IS_APEX_AMP
                        Using NVIDIA APEX Automatic Mixed Precision (AMP
  --amp_opt_level AMP_OPT_LEVEL
                        optimization level of apex amp.
```


After you set all the arguments properly, you can simply `cd` to `splitnet/script`  and run
```
./train_split.sh
```
to start training.

### Monitoring the training process through tensorboard

```
tensorboard --logdir=your_logdir --port=your_port

# or run script/tensorboard.sh
```

![img_tensorboard](miscs/fig3_tensorboard.png)

### GPU memory usage

For cifar, with mixed precision training,
1 RTX 2080Ti (11GB) is enough to conduct most experiments.
For ImageNet, with mixed precision training,
8 RTX 2080Ti (11Gb) or 4 Tesla V100 (32GB) is enough to conduct most experiments. 

## Evaluation

See `script/eval.sh` or `script/test.sh` for detailed information.

You should also specify some variables in the scripts.

- `data`, where you save your dataset.

- `resume`, where your checkpoints locate.

Then run 
```
./eval.sh
```

* You can run `script/summary.sh` to get the number of parameters and FLOPs of a model.


## Citations

```
@misc{2020_splitnet,
  author =   {Shuai Zhao and Liguang Zhou and Wenxiao Wang and Deng Cai and Tin Lun Lam and Yangsheng Xu},
  title =    {Towards Better Accuracy-efficiency Trade-offs: Divide and Co-training},
  howpublished = {arXiv},
  year = {2020}
}
```


## Licenses
Most of the code here is licensed Apache 2.0.
However, this repo contains much third party code.
It is your responsibility to ensure you comply with license
here and conditions of any dependent licenses


As for the core code of splitnet (i.e., code about dividing the model and co-training)
and the pretrained models, they are under the CC-BY-NC 4.0 license.
See [LICENSE](miscs/LICENSE) for additional details.
Hope you can understand this because this work is funded by a for-profit company.


## Acknowledgements

<!--ts-->
* [pytorch/vision](https://github.com/pytorch/vision)
* [NVIDIA/apex](https://github.com/NVIDIA/apex)
* [rwightman/pytorch-image-models](https://github.com/rwightman/pytorch-image-models)
* [Cadene/pretrained-models.pytorch](https://github.com/Cadene/pretrained-models.pytorch)
* [ansleliu/EfficientNet.PyTorch](https://github.com/ansleliu/EfficientNet.PyTorch)
* [lukemelas/EfficientNet-PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch)
* [tensorflow/tpu/models/official/efficientnet/](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)
* [ecs-vlc/FMix](https://github.com/ecs-vlc/FMix)
* [BIGBALLON/CIFAR-ZOO](https://github.com/BIGBALLON/CIFAR-ZOO)
* [kakaobrain/fast-autoaugment](https://github.com/kakaobrain/fast-autoaugment)
* [hongyi-zhang/mixup](https://github.com/hongyi-zhang/mixup)
* [DeepVoltaire/AutoAugment](https://github.com/DeepVoltaire/AutoAugment)
* [uoguelph-mlrg/Cutout](https://github.com/uoguelph-mlrg/Cutout)
* [zhanghang1989/PyTorch-Encoding](https://github.com/zhanghang1989/PyTorch-Encoding)
* [zhanghang1989/ResNeSt](https://github.com/zhanghang1989/ResNeSt)
* [PistonY/torch-toolbox](https://github.com/PistonY/torch-toolbox)
* [Lyken17/pytorch-OpCounter](https://github.com/Lyken17/pytorch-OpCounter)
* [Cerebras/online-normalization](https://github.com/Cerebras/online-normalization)
* [maciejczyzewski/batchboost](https://github.com/maciejczyzewski/batchboost)
* [ZJULearning/RMI](https://github.com/ZJULearning/RMI)
<!--te-->

![img_cad](miscs/zju_cad.jpg)
![img_cuhksz](miscs/cuhksz.png)
![img_airs](miscs/akg_airs.png)

