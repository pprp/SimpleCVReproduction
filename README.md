# SimpleCVReproduction

一些论文复现，比如Attention 模块。

如果模块过多，则使用多个python模块进行构建，尽量不新建文件夹

将感兴趣/推荐的模型也放在这个库中，以供学习。

尽量提供简化版本的，便于理解的模型文件。

- Simple_CenterNet 是一个简化版本的，正在试验中。
- SmallObjectAugmentation是一个专门用于小目标增强库，实际效果不是很理想。增加了一些处理工具模块。
- Plug-and-play module: 即插即用模块：
  - attention 实现或者复制官方的pytorch实现，即插即用的注意力模块。
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
- DBFace:readme中展示了非常好的检测效果碾压retinaFace,CenterFace，目前只提供inference，还没有train，期待公开训练代码...
- simple_keypoints: 简单的关键点检测
- ultralytics_newest_yolov3: 这个库在coco数据集上已经刷到了SOTA，但是根据我在2020年4月14日跑的自己的数据集来说，效果并不好，即便加载预训练权重，yolov3.cfg只能达到60%的mAP, 可能是作者调用了大量的trick来对coco上的结果进行优化，虽然在COCO上mAP@0.5都刷到62.8了，但是训练自己的数据集效果却越来越差。之前也用过这个训练同样的数据集，老版本的这个库虽然在coco上效果不那么惊人，但是在我的数据集上能达到80%的mAP。不知道问题在何，如果有看到这里的大佬欢迎在issue中交流一下，指点一下。