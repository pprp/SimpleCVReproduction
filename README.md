# SimpleCVReproduction

我将感兴趣/推荐的模型也放在这个库中，以供学习。由于好多库从头开始学习难度太大，所以在这里提供了笔者的部分注释，其中大部分都是跑过的模型、准备读的代码、已经读过的代码笔记、自己开发的simple系列简单代码、常用代码段。

尽量提供简化版本的，便于理解的模型文件。

如果有推荐的便于初学者学习的库，也欢迎在issue中提出，之后会补充上来。

## 即插即用模块（注意力模块）

> PS: 在GiantPandaCV公众号后台回复即插即用可以得到一份简单的指导手册。

注意力模块Attention & 即插即用模块 现在已经迁移到[awesome-attention-mechanism-in-cv](https://github.com/pprp/awesome-attention-mechanism-in-cv)欢迎关注

> PS: 关于如何在YOLOv3中加入以上模块，可以访问[这个博客](https://blog.csdn.net/DD_PP_JJ/article/details/104109369)，这个里边实现了SE,SK,CBAM,SPP,ASPP等在内的模型，对应的代码在https://github.com/GiantPandaCV/yolov3-point。

## 其他推荐项目

- CenterNet 是一个简化版本的（并非原版），正在分析和学习源码。
- SmallObjectAugmentation是一个专门用于小目标增强库，实际效果不是很理想。增加了一些处理工具模块。
- captcha-CTC-loss CTC loss+ LSTM 
- deep_sort-master 官方实现，通过该库理解了标准的输入输出格式。
- easy-receptive-fields-pytorch-master: 用于计算pytorch常用CNN的感受野，非常方便
- kalman 知乎上的一个简单的卡尔曼滤波算法实现代码
- opencv-mot 用OpenCV中自带的跟踪器如KCF等实现跟踪，第一帧目标需要在代码中指定。
- pytorch-commen-code pytorch中常用的一些代码
- pytorch-grad-cam-master grad cam的实现
- pytorch-semseg pytorch实现语义分割，目前仅在自己数据集上训练了Unet，无法收敛。
- siamese-triplet : 孪生网络+triplet loss
- simple-DCGAN : DCGAN, 还没来得及研究
- simple-faster-rcnn-pytorch 陈云老师的实现
- simple-triple-loss 自己仿照一个库写了一个简化版的triple loss
- tiny_classifier : 目标检测级联一个分类网络中的分类网络的简单实现。
- tools: 目前只有voc2coco.py工具
- yolov3-6: U版yolov3中release出来的稳定版本，其中使用的是原始的yolov3 loss，改动不多。
- DBFace:readme中展示了非常好的检测效果碾压retinaFace,CenterFace，目前只提供inference，还没有train，期待公开训练代码...（ps: landmark用的是heatmap）
- simple_keypoints: 简单的关键点检测，提供了通过heatmap和回归两种方法进行检测
- YOLOv3-complete-pruning: 基于U版进行剪枝的库，效果还不错。
- yolov5: 于8月6日，在小麦检测比赛上效果惊人，超过了cascade rcnn等传统比赛常用模型。
- TSNE: 可视化重识别数据集、分类数据集，效果挺好的，不过一般需要对服务器内存要求比较高
- tikz-cnn: 用LaTeX中tikz包绘制卷积神经网络结构图
- nni库：AutoML中比较好用的库
- R-CenterNet：林亿大佬写的可旋转目标检测框架，极简风格，便于快速掌握，很赞，打算拜读一下。[最新版代码点这里](https://github.com/ZeroE04/R-CenterNet)
- cvtranforms: 可以替代pytorch中transform(PIL-based)，据说让数据读取快三倍。 ps:另外一个可选方案是albumentations包，提供pip安装。
- PyTorch-Lightning: 类似keras一样的封装包，可以快速开发迭代。但是有一定门槛，需要深入实现的时候，会遇到麻烦。可以参考其设计理念，自己设计包的架构。
- VisionTransformer: 极简源码实现vit, [原链接在这里](https://github.com/lucidrains/vit-pytorch)


## 感谢

@zhongqiu1245 补充的borderDet中的BAM模块,补充了FPT

@1187697147 补充的context-gating模块

@cmsfw-github 指出了simple_keypoint中的bug

@1187697147 建议更新了AFF和iAFF模块源码
