# SimpleCVReproduction

![](logo.png)

将感兴趣/推荐的模型也放在这个库中，以供学习。由于好多库从头开始学习难度太大，在这里提供了笔者的部分注释，其中大部分都是跑过的模型、准备读的代码、已经读过的代码笔记、以及开发的simple系列简单代码、常用代码段等。

本项目致力于提供简化版本的，便于理解的模型文件。

如果有推荐的便于初学者学习的库，也欢迎在issue中提出和补充。

本项目大部分内容是来源于Github，不会用做商业用途，如有侵权，请联系笔者删除。

## 目录

- [即插即用模块&注意力模块](即插即用模块&注意力模块)
- [项目推荐](项目推荐)
- [致谢](致谢)
- [贡献](贡献)


## 即插即用模块&注意力模块

原项目已经迁移至新的地址：[Awesome-Attention-Mechanism-in-cv](https://github.com/pprp/awesome-attention-mechanism-in-cv)

主要内容包括:

- 计算机视觉领域中**注意力**模块。
- 计算机视觉中**即插即用**模块。[code](https://github.com/pprp/SimpleCVReproduction/tree/master/Plug-and-play%20module)
- **Vision Transformer**系列工作。

更多介绍：

- [我们是如何结合注意力机制改进YOLOv3进行目标检测?](https://zhuanlan.zhihu.com/p/231168560) [Code](https://github.com/GiantPandaCV/yolov3-point)
- [如何在YOLOv3中加入注意力模块or即插即用模块](https://blog.csdn.net/DD_PP_JJ/article/details/104109369)
- [神经网络加上注意力机制，精度反而下降，为什么会这样呢？](https://www.zhihu.com/question/478301531/answer/2280232845)
- [CNN、Transformer、MLP架构经验性分析](https://zhuanlan.zhihu.com/p/449280021)
- [CV中的注意力机制之ShuffleAttention](https://zhuanlan.zhihu.com/p/350912960)
- [CV中的注意力机制之并联版的CBAM-BAM模块](https://zhuanlan.zhihu.com/p/102033063)
- [CV中的注意力机制之SKNet-SENet的提升版](https://zhuanlan.zhihu.com/p/102034839)

- [CV中的注意力机制之简单而有效的CBAM模块](https://zhuanlan.zhihu.com/p/102035273)

- [CV中的注意力机制之SENet中的SE模块](https://zhuanlan.zhihu.com/p/102035721)

- [CV中的注意力机制之语义分割中的scSE模块](https://zhuanlan.zhihu.com/p/102036086)

- [CV中的注意力机制之Non-Local Network的理解与实现](https://zhuanlan.zhihu.com/p/102984842)
- [CV中的注意力机制之融合Non-Local和SENet的GCNet](https://zhuanlan.zhihu.com/p/102990363)
- [CV中的注意力机制之BiSeNet中的FFM模块与ARM模块](https://zhuanlan.zhihu.com/p/105925132)

## 项目推荐

| 项目                          | 介绍                                                         | 链接                                                         |
| ----------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| CenterNet                     | 简化版本的CenterNet目标检测算法(第三方实现)                  | [link](https://github.com/pprp/SimpleCVReproduction/tree/master/CenterNet) |
| SmallObjectAugmentation       | 针对小目标进行数据增强库,在笔者数据集效果不理想              | [link](https://github.com/pprp/52RL)                         |
| DBFace                        | 实时单阶段人脸检测器                                         | [link](https://github.com/pprp/SimpleCVReproduction/tree/master/DBFace) |
| DarkLabel                     | 专门用于[DarkLabel](https://zhuanlan.zhihu.com/p/141036498)软件转化的系列脚本 | [link](https://github.com/pprp/SimpleCVReproduction/tree/master/DarkLabel) |
| Latex/latex_algo              | 用latex写的伪代码示例                                        | [link](https://github.com/pprp/SimpleCVReproduction/tree/master/Latex/latex_algo) |
| MLP                           | MLP-Mixer,ResMLP,RepMLP简单源码                              | [link](https://github.com/pprp/SimpleCVReproduction/tree/master/MLP) |
| NAS                           | 感兴趣的神经网络结构搜索算法                                 | [link](https://github.com/pprp/SimpleCVReproduction/tree/master/NAS) |
| Plug-and-play Module          | 即插即用模块                                                 | [link](https://github.com/pprp/SimpleCVReproduction/tree/master/Plug-and-play%20module) |
| PyTorch-Lightning             | Lightning的基础使用案例                                      | [link](https://github.com/pprp/SimpleCVReproduction/tree/master/PyTorch-Lightning) |
| 52RL                          | 参加DataWhale深度强化学习课程代码 [code](https://github.com/pprp/52RL) | [link](https://github.com/pprp/52RL)                         |
| Vision Transformer            | 最经典的ViT实现, 训练代码在[code](https://github.com/pprp/pytorch-cifar-model-zoo) | [link](https://github.com/pprp/SimpleCVReproduction/tree/master/VisionTransformer) |
| YOLOv3-complete-pruning       | YOLOv3经典的剪枝算法                                         | [link](https://github.com/pprp/SimpleCVReproduction/tree/master/YOLOv3-complete-pruning) |
| captcha-CTC-loss              | CTC loss+ LSTM                                               | [link](https://github.com/pprp/SimpleCVReproduction/tree/master/captcha-CTC-loss) |
| cifarTrick                    | 原先收集的部分Trick更多Trick在[Tricks](https://github.com/pprp/pytorch-cifar-model-zoo) | [link](https://github.com/pprp/SimpleCVReproduction/tree/master/cifarTrick) |
| cvtransforms                  | 数据增强方法, 可以替代pytorch中transform(PIL-based)，据说让数据读取快三倍 | [link](https://github.com/pprp/SimpleCVReproduction/tree/master/cvtransforms) |
| deep_sort                     | 官方实现的DeepSort算法                                       | [link](https://github.com/pprp/SimpleCVReproduction/tree/master/deep_sort) |
| deep_sort_yolov3_pytorch      | 笔者自己实现和改进的DeepSort算法                             | [link](https://github.com/pprp/deep_sort_yolov3_pytorch)     |
| easy-receptive-fields-pytorch | 感受野计算                                                   | [link](https://github.com/pprp/SimpleCVReproduction/tree/master/easy-receptive-fields-pytorch) |
| fine_grained_baseline         | 细粒度识别baseline，Bilinear Pooling操作                     | [link](https://github.com/pprp/SimpleCVReproduction/tree/master/fine_grained_baseline) |
| flask-yolo                    | flask配合yolo算法实现网页                                    | [link](https://github.com/pprp/SimpleCVReproduction/tree/master/flask-yolo) |
| kalman                        | 卡尔曼滤波实现与测试                                         | [link](https://github.com/pprp/SimpleCVReproduction/tree/master/kalman) |
| libfacedetection.train        | 人脸检测训练代码                                             | [link](https://github.com/pprp/SimpleCVReproduction/tree/master/kalman) |
| opencv-mot                    | 使用Opencv实现多目标跟踪                                     | [link](https://github.com/pprp/SimpleCVReproduction/tree/master/opencv-mot) |
| pandoc-starter                | Pandoc是Markdown转化器，很方便                               | [link](https://github.com/pprp/SimpleCVReproduction/tree/master/pandoc-starter) |
| pytorch-commen-code           | 常用的pytorch代码片段                                        | [link](https://github.com/pprp/SimpleCVReproduction/tree/master/pandoc-starter) |
| pytorch-grad-cam              | Grad Cam代码实现，特征图可视化                               | [link](https://github.com/pprp/SimpleCVReproduction/tree/master/pytorch-grad-cam-master) |
| pytorch-semseg                | 语义分割代码库收集，经测试无法收敛(私有数据集)               | [link](https://github.com/pprp/SimpleCVReproduction/tree/master/pytorch-semseg) |
| siamese-triplet               | 孪生网络+Triplet Loss实现                                    | [link](https://github.com/pprp/SimpleCVReproduction/tree/master/simple-faster-rcnn-pytorch) |
| simple-faster-rcnn-pytorch    | 陈云老师的faster rcnn实现                                    | [link](https://github.com/pprp/SimpleCVReproduction/tree/master/simple-faster-rcnn-pytorch) |
| simple-triple-loss            | 笔者自己实现的triplet loss                                   | [link](https://github.com/pprp/SimpleCVReproduction/tree/master/simple-triple-loss) |
| simple_keypoint               | **[推荐]** 笔者极简代码实现关键点识别，提供根据heatmap进行识别的方法 | [link](https://github.com/pprp/SimpleCVReproduction/tree/master/simple_keypoint) |
| tikz_cnn                      | 使用latex绘制CNN图                                           | [link](https://github.com/pprp/SimpleCVReproduction/tree/master/tikz_cnn) |
| tsne                          | tsne可视化数据集                                             | [link](https://github.com/pprp/SimpleCVReproduction/tree/master/tsne) |
| tools                         | voc2coco脚本，yolo anchor聚类脚本                            | [link](https://github.com/pprp/SimpleCVReproduction/tree/master/tools) |
| tiny_classifier               | 超级简单的分类代码+focal loss使用                            | [link](https://github.com/pprp/SimpleCVReproduction/tree/master/tiny_classifier) |
| yolov3-6                      | 第六次release版本，属于老版本yolo实现                        | [link](https://github.com/pprp/SimpleCVReproduction/tree/master/yolov3-6) |


## 致谢

@zhongqiu1245 补充的borderDet中的BAM模块,补充了FPT

@1187697147 补充的context-gating模块

@cmsfw-github 指出了simple_keypoint中的bug

@1187697147 建议更新了AFF和iAFF模块源码

## 贡献

欢迎在issue中提出补充推荐的项目。

欢迎关注“GiantPandaCV”公众号以及“神经网络架构搜索”公众号查看相关博客。

