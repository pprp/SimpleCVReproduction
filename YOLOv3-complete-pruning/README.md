# YOLOv3-complete-pruning

本项目以[ultralytics/yolov3](https://github.com/ultralytics/yolov3)为YOLOv3的Pytorch实现，并在[YOLOv3-model-pruning](https://github.com/Lam1360/YOLOv3-model-pruning)剪枝的基础上，推出了4个YOLO-v3剪枝版本。(在此致谢两位)

此外，最近还更新了YOLO的1bit、4bit、8bit、16bit量化。

|剪枝方式|<center>优点</center>|<center>缺点</center> |
| --- | --- | --- |
| 正常剪枝 |不对shortcut剪枝，拥有可观且稳定的压缩率，无需微调。  |压缩率达不到极致。  |
| 极限剪枝 |极高的压缩率。  |需要微调。  |
| 规整剪枝 |专为硬件部署设计，剪枝后filter个数均为8的倍数，无需微调。 | 为规整牺牲了部分压缩率。 |
| Tiny剪枝 |稳定的压缩率。  |由于Tiny本来已很小，压缩率中规中矩。  |

## 项目特点

1.采用的YOLO-v3实现较为准确，mAP相对较高。

模型        | 320         | 416         | 608
---             | ---         | ---         | ---
`YOLOv3`        | 51.8 (51.5) | 55.4 (55.3) | 58.2 (57.9)
`YOLOv3-tiny`   | 29.0        | 32.9 (33.1) | 35.5

2.提供对YOLOv3及Tiny的多种剪枝版本、量化版本，以适应不同的需求。

3.剪枝后保存为.weights格式，可在任何框架下继续训练、推理，或以图像视频展示。

<img src="https://pjreddie.com/media/image/Screen_Shot_2018-03-24_at_10.48.42_PM.png" width="500">

4.目前支持情况

|<center>剪枝方式</center>|<center>单卡</center>|<center>多卡</center>|   
| --- | --- | --- |
|<center>正常训练</center>|<center>√</center>|<center>√</center>|
|<center>稀疏化</center>|<center>√</center>|<center>√</center>  |
|<center>正常剪枝</center>|<center>√</center>|<center>√</center>|
|<center>规整剪枝</center>  | <center>√</center> |<center>√</center>  |
|<center>极限剪枝(shortcut)</center>  | <center>√</center> | <center>√</center> |
|<center>Tiny剪枝</center>|<center>√</center>|<center>√</center>  |

|<center>二值量化</center>|<center>8bit量化</center>|<center>16bit量化</center>|<center>混合量化</center>|<center>任意bit量化</center>|
| --- | --- | --- | --- | --- |
|<center>√</center>|<center>√</center>|<center>√</center>|<center>√</center>|<center>√</center>|


## 最新进展

- 2019年12月26日：支持任意bit量化。
- 2019年11月6日：极限剪枝已支持无需微调。
- 非常感谢github大佬[tanluren](https://github.com/tanluren)，对该项目指出的众多问题和支持，实在太强了^_^。


## 环境搭建

1.由于采用[ultralytics/yolov3](https://github.com/ultralytics/yolov3)的YOLO实现，环境搭建见[ultralytics/yolov3](https://github.com/ultralytics/yolov3)。这里重复介绍一下：

- `numpy`
- `torch >= 1.1.0`
- `opencv-python`
- `tqdm`

可直接`pip3 install -U -r requirements.txt`搭建环境，或根据该.txt文件使用conda搭建。

## 数据获取

依然采用oxford hand数据集

<img src="https://www.robots.ox.ac.uk/~vgg/data/hands/hand_dataset_files/Picture_1.png" width="500">


1.下载[数据集](http://www.robots.ox.ac.uk/~vgg/data/hands/downloads/hand_dataset.tar.gz)，并解压至/data目录下，得到hand_dataset文件夹。

2.执行`python converter.py` ，生成 images、labels 文件夹和 train.txt、valid.txt 文件。

3.获取YOLO预训练权重，/weights文件夹下执行`bash download_yolov3_weights.sh`，或自行下载。

4.至此，数据部分完成。

## 剪枝篇

1.正常训练

```bash
python3 train.py --data data/oxfordhand.data --batch-size 32 --accumulate 1 --weights weights/yolov3.weights --cfg cfg/yolov3-hand.cfg
```

2.稀疏化训练

`-sr`开启稀疏化，`--s`指定稀疏因子大小，`--prune`指定稀疏类型。

其中：

`--prune 0`为正常剪枝和规整剪枝的稀疏化

`--prune 1`为极限剪枝的稀疏化

`--prune 2`为Tiny剪枝的稀疏化

```bash
python3 train.py --data data/oxfordhand.data --batch-size 32 --accumulate 1 --weights weights/yolov3.weights --cfg cfg/yolov3-hand.cfg -sr --s 0.001 --prune 0 
```

3.模型剪枝

- 正常剪枝
```bash
python3 normal_prune.py
```
- 规整剪枝
```bash
python3 regular_prune.py
```
- 极限剪枝
```bash
python3 shortcut_prune.py
```
- Tiny剪枝
```bash
python3 prune_tiny_yolo.py
```
需要注意的是，这里需要在.py文件内，将opt内的cfg和weights变量指向第2步稀疏化后生成的cfg文件和weights文件。
此外，可通过增大代码中percent的值来获得更大的压缩率。（若稀疏化不到位，且percent值过大，程序会报错。）

## 量化篇

1 指定需要量化的层

打开任意一个可用的配置文件，例如yolov3-hand.cfg。将需要量化的层，从原来的convolutional替换为quantize_convolutional。

2 指定量化方式

通过修改models.py中的W_bit和A_bit，指定权重的量化方式，激活的量化方式。（目前默认为16bit量化）

3 量化训练


```bash
python3 train.py --data data/oxfordhand.data --batch-size 32 --accumulate 1 --weights weights/yolov3.weights --cfg cfg/yolov3-quantize-hand.cfg
```

与正常训练相同，只是cfg要指向修改过的cfg文件。

## 推理展示

这里，我们不仅可以使用原始的YOLOV3用来推理展示，还可使用我们剪枝后的模型来推理展示。（修改cfg，weights的指向即可）

<img src="https://user-images.githubusercontent.com/26833433/64067835-51d5b500-cc2f-11e9-982e-843f7f9a6ea2.jpg" width="500">

```bash
python3 detect.py --source ...
```

- Image:  `--source file.jpg`
- Video:  `--source file.mp4`
- Directory:  `--source dir/`
- Webcam:  `--source 0`
- RTSP stream:  `--source rtsp://170.93.143.139/rtplive/470011e600ef003a004ee33696235daa`
- HTTP stream:  `--source http://wmccpinetop.axiscam.net/mjpg/video.mjpg`



例如：
```bash
python3 detect.py --cfg cfg/prune_0.8_yolov3-hand.cfg --weights weights/yolov3_hand_pruning_percent0.8.weights --data data/oxfordhand.data --source test.jpg
```

## 剪枝效果

以下数据除极限剪枝外，均未微调。

### YOLO-v3剪枝

| 模型 | 参数量 |模型体积  |压缩率  |耗时  |mAP  |
| --- | --- | --- | --- | --- | --- |
| Baseline(416)| 61.5M |246.4MB  |0%  |11.7ms  |0.7924  |
| 正常剪枝 | 10.9M |43.9MB  |82.2%  |5.92ms  |0.7712  |
| 规整剪枝 | 15.31M |61.4MB  |75.1%  |6.01ms  |0.7832  |
| 极限剪枝 | 7.13M |28.6MB  |88.4%  |5.90ms  |0.7382  |

### YOLO-v3-Tiny剪枝

| 模型 |参数量  | 模型体积 |  压缩率| 耗时 | mAP |
| --- | --- | --- | --- | --- | --- |
| Baseline(416) | 8.7M | 33.1MB | 0% | 2.2ms | 0.6378 |
| Tiny剪枝 | 4.4M | 16.8MB | 40.1% |  2.0ms| 0.6132 |

## 量化效果（数据更新中）

以下量化数据中，权重激活均被量化。

### YOLO-v3量化

| 模型 |mAP  |
| --- | --- |
| Baseline(416)|0.8246  |
| 第一层float32+中间层8bit+最后一层float32 |0.8174  |
| 全16bit量化 |0.8132  |
| 全8bit量化 | 0.8024  |



## 核心思想

剪枝方法来源于论文[Learning Efficient Convolutional Networks through Network Slimming](https://arxiv.org/abs/1708.06519)，剪枝无需微调方法来源于[Rethinking the Smaller-Norm-Less-Informative Assumption in Channel Pruning of Convolution Layers](https://arxiv.org/abs/1802.00124?context=cs)。


量化方法来源于论文[DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients](https://arxiv.org/abs/1606.06160)。

此外，具体实现时，在论文作者的基础上做了改进。


## 互动

### 1.如何获得较高的压缩率？

提高压缩率的关键在于稀疏化训练，可以加大`--s`的值并迭代训练多次等手段。

### 2.我的压缩率比表格中更高！

以上数据仅仅是测试了不到20次的结果，如果有同学的压缩率更高，欢迎在评论区分享！

### 3.程序报错怎么办？

#### YOLOv3报错
由于采用了[ultralytics/yolov3](https://github.com/ultralytics/yolov3)为YOLOv3的Pytorch实现，因此这类错误可跳转至此链接询问。

#### 剪枝量化错误

一定要在本评论区留言，我会尽快修正！



