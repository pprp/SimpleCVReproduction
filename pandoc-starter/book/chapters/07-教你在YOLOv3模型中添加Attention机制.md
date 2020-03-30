# attention

> 前言：【从零开始学习YOLOv3】系列越写越多，本来安排的内容比较少，但是在阅读代码的过程中慢慢发掘了一些新的亮点，所以不断加入到这个系列中。之前都在读YOLOv3中的代码，已经学习了cfg文件、模型构建等内容。本文在之前的基础上，对模型的代码进行修改，将之前Attention系列中的SE模块和CBAM模块集成到YOLOv3中。

## 1. 规定格式

正如`[convolutional]`,`[maxpool]`,`[net]`,`[route]`等层在cfg中的定义一样，我们再添加全新的模块的时候，要规定一下cfg的格式。做出以下规定：

在SE模块（具体讲解见: [【cv中的Attention机制】最简单最易实现的SE模块](<https://mp.weixin.qq.com/s?__biz=MzA4MjY4NTk0NQ==&mid=2247484504&idx=2&sn=3aa20e4a80da1e2125673296e29b4217&chksm=9f80becea8f737d80ec11d49d0f9172f259c2f33ff75a2ce9542b18ee6f5bac8732490946f51&token=897871599&lang=zh_CN#rd>)）中，有一个参数为`reduction`,这个参数默认是16，所以在这个模块中的详细参数我们按照以下内容进行设置：

```python
[se]
reduction=16
```

在CBAM模块（具体讲解见: [【CV中的Attention机制】ECCV 2018 Convolutional Block Attention Module](<https://mp.weixin.qq.com/s?__biz=MzA4MjY4NTk0NQ==&mid=2247484531&idx=1&sn=625065862b28608428acb21da3330717&chksm=9f80bee5a8f737f399f0f564883337154dd8ca3ad5c246c85a86a88b0ac8ede7bf59ffc04554&token=897871599&lang=zh_CN#rd>)）中，空间注意力机制和通道注意力机制中一共存在两个参数：`ratio`和`kernel_size`, 所以这样规定CBAM在cfg文件中的格式：

```python
[cbam]
ratio=16
kernelsize=7
```

## 2. 修改解析部分

由于我们添加的这些参数都是自定义的，所以需要修改解析cfg文件的函数，之前讲过，需要修改`parse_config.py`中的部分内容：

```python
def parse_model_cfg(path):
    # path参数为: cfg/yolov3-tiny.cfg
    if not path.endswith('.cfg'):
        path += '.cfg'
    if not os.path.exists(path) and \
    	   os.path.exists('cfg' + os.sep + path):
        path = 'cfg' + os.sep + path

    with open(path, 'r') as f:
        lines = f.read().split('\n')

    # 去除以#开头的，属于注释部分的内容
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines]
    mdefs = []  # 模块的定义
    for line in lines:
        if line.startswith('['):  # 标志着一个模块的开始
            '''
            eg:
            [shortcut]
            from=-3
            activation=linear
            '''
            mdefs.append({})
            mdefs[-1]['type'] = line[1:-1].rstrip()
            if mdefs[-1]['type'] == 'convolutional':
                mdefs[-1]['batch_normalize'] = 0 
        else:
            key, val = line.split("=")
            key = key.rstrip()

            if 'anchors' in key:
                mdefs[-1][key] = np.array([float(x) for x in val.split(',')]).reshape((-1, 2))
            else:
                mdefs[-1][key] = val.strip()

    # Check all fields are supported
    supported = ['type', 'batch_normalize', 'filters', 'size',\
                 'stride', 'pad', 'activation', 'layers', \
                 'groups','from', 'mask', 'anchors', \
                 'classes', 'num', 'jitter', 'ignore_thresh',\
                 'truth_thresh', 'random',\
                 'stride_x', 'stride_y']

    f = []  # fields
    for x in mdefs[1:]:
        [f.append(k) for k in x if k not in f]
    u = [x for x in f if x not in supported]  # unsupported fields
    assert not any(u), "Unsupported fields %s in %s. See https://github.com/ultralytics/yolov3/issues/631" % (u, path)

    return mdefs
```

以上内容中，需要改的是supported中的字段，将我们的内容添加进去：

```python
supported = ['type', 'batch_normalize', 'filters', 'size',\
            'stride', 'pad', 'activation', 'layers', \
            'groups','from', 'mask', 'anchors', \
            'classes', 'num', 'jitter', 'ignore_thresh',\
            'truth_thresh', 'random',\
            'stride_x', 'stride_y',\
            'ratio', 'reduction', 'kernelsize']
```

## 3. 实现SE和CBAM

具体原理还请见[【cv中的Attention机制】最简单最易实现的SE模块](<https://mp.weixin.qq.com/s?__biz=MzA4MjY4NTk0NQ==&mid=2247484504&idx=2&sn=3aa20e4a80da1e2125673296e29b4217&chksm=9f80becea8f737d80ec11d49d0f9172f259c2f33ff75a2ce9542b18ee6f5bac8732490946f51&token=897871599&lang=zh_CN#rd>)和[【CV中的Attention机制】ECCV 2018 Convolutional Block Attention Module](<https://mp.weixin.qq.com/s?__biz=MzA4MjY4NTk0NQ==&mid=2247484531&idx=1&sn=625065862b28608428acb21da3330717&chksm=9f80bee5a8f737f399f0f564883337154dd8ca3ad5c246c85a86a88b0ac8ede7bf59ffc04554&token=897871599&lang=zh_CN#rd>)这两篇文章，下边直接使用以上两篇文章中的代码：

**SE**

```python
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
```

**CBAM**

```python
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3,7), "kernel size must be 3 or 7"
        padding = 3if kernel_size == 7else1

        self.conv = nn.Conv2d(2,1,kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)
    
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, rotio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False), nn.ReLU(),
            nn.Conv2d(in_planes // rotio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)
```

以上就是两个模块的代码，添加到`models.py`文件中。

## 4. 设计cfg文件

这里以`yolov3-tiny.cfg`为baseline，然后添加注意力机制模块。

CBAM与SE类似，所以以SE为例，添加到backbone之后的部分，进行信息重构(refinement)。

```python
[net]
# Testing
batch=1
subdivisions=1
# Training
# batch=64
# subdivisions=2
width=416
height=416
channels=3
momentum=0.9
decay=0.0005
angle=0
saturation = 1.5
exposure = 1.5
hue=.1

learning_rate=0.001
burn_in=1000
max_batches = 500200
policy=steps
steps=400000,450000
scales=.1,.1

[convolutional]
batch_normalize=1
filters=16
size=3
stride=1
pad=1
activation=leaky

[maxpool]
size=2
stride=2

[convolutional]
batch_normalize=1
filters=32
size=3
stride=1
pad=1
activation=leaky

[maxpool]
size=2
stride=2

[convolutional]
batch_normalize=1
filters=64
size=3
stride=1
pad=1
activation=leaky

[maxpool]
size=2
stride=2

[convolutional]
batch_normalize=1
filters=128
size=3
stride=1
pad=1
activation=leaky

[maxpool]
size=2
stride=2

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[maxpool]
size=2
stride=2

[convolutional]
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

[maxpool]
size=2
stride=1

[convolutional]
batch_normalize=1
filters=1024
size=3
stride=1
pad=1
activation=leaky

[se]
reduction=16

# 在backbone结束的地方添加se模块
#####backbone######

[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

[convolutional]
size=1
stride=1
pad=1
filters=18
activation=linear



[yolo]
mask = 3,4,5
anchors = 10,14,  23,27,  37,58,  81,82,  135,169,  344,319
classes=1
num=6
jitter=.3
ignore_thresh = .7
truth_thresh = 1
random=1

[route]
layers = -4

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[upsample]
stride=2

[route]
layers = -1, 8

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[convolutional]
size=1
stride=1
pad=1
filters=18
activation=linear

[yolo]
mask = 0,1,2
anchors = 10,14,  23,27,  37,58,  81,82,  135,169,  344,319
classes=1
num=6
jitter=.3
ignore_thresh = .7
truth_thresh = 1
random=1
```

## 5. 模型构建

以上都是准备工作，以SE为例，我们修改`model.py`文件中的模型加载部分，并修改forward函数部分的代码，让其正常发挥作用：

在`model.py`中的`create_modules`函数中进行添加：

```python
        elif mdef['type'] == 'se':
            modules.add_module(
                'se_module',
                SELayer(output_filters[-1], reduction=int(mdef['reduction'])))
```

然后修改Darknet中的forward部分的函数：

```python
def forward(self, x, var=None):
    img_size = x.shape[-2:]
    layer_outputs = []
    output = []

    for i, (mdef,
            module) in enumerate(zip(self.module_defs, self.module_list)):
        mtype = mdef['type']
        if mtype in ['convolutional', 'upsample', 'maxpool']:
            x = module(x)
        elif mtype == 'route':
            layers = [int(x) for x in mdef['layers'].split(',')]
            if len(layers) == 1:
                x = layer_outputs[layers[0]]
            else:
                try:
                    x = torch.cat([layer_outputs[i] for i in layers], 1)
                except:  # apply stride 2 for darknet reorg layer
                    layer_outputs[layers[1]] = F.interpolate(
                        layer_outputs[layers[1]], scale_factor=[0.5, 0.5])
                    x = torch.cat([layer_outputs[i] for i in layers], 1)

        elif mtype == 'shortcut':
            x = x + layer_outputs[int(mdef['from'])]
        elif mtype == 'yolo':
            output.append(module(x, img_size))
        layer_outputs.append(x if i in self.routs else [])
```

在forward中加入SE模块，其实很简单。SE模块与卷积层，上采样，最大池化层地位是一样的，不需要更多操作，只需要将以上部分代码进行修改：

```python
    for i, (mdef,
            module) in enumerate(zip(self.module_defs, self.module_list)):
        mtype = mdef['type']
        if mtype in ['convolutional', 'upsample', 'maxpool', 'se']:
            x = module(x)
```

CBAM的整体过程类似，可以自己尝试一下，顺便熟悉一下YOLOv3的整体流程。

> 后记：本文的内容很简单，只是添加了注意力模块，很容易实现。不过具体注意力机制的位置、放多少个模块等都需要做实验来验证。注意力机制并不是万金油，需要多调参，多尝试才能得到满意的结果。欢迎大家联系我加入群聊，反馈在各自数据集上的效果。
>
> ps: 最近大家注意身体，出门戴口罩。