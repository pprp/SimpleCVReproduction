---
title: "从零开始学习YOLOv3系列合集"
author: [GiantPandaCV-逍遥王可爱]
date: "2020-06-14"
subject: "Markdown"
keywords: [YOLOv3, 教程, GiantPandaCV]
subtitle: "GiantPandaCV公众号"
titlepage: true
titlepage-text-color: "000000"
titlepage-background: "backgrounds/background6.pdf"
---

# 第一部分：Darknet构建

一个yolov3的pytorch版快速实现工程的见github：

https://github.com/ayooshkathuria/YOLO_v3_tutorial_from_scratch 

该工程的作者写了个入门教程，在这个教程中，作者使用 PyTorch 实现基于 YOLO v3 的目标检测器，该教程一共有五个部分，虽然并没有含有训练部分。链接：

https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/

这个教程已经有了全的翻译版本，分为上下两个部分，上部分的链接：

https://www.jiqizhixin.com/articles/2018-04-23-3

下部分的链接：[从零开始PyTorch项目：YOLO v3目标检测实现](https://www.jiqizhixin.com/articles/042602?from=synced&keyword=从零开始PyTorch项目：YOLO v3目标检测实现)

有了上面这些教程，我自然不会重复之前的工作，而是给出每个程序**每行代码最详细**全面的小白入门注释，不论基础多差都能看懂，注释到**每个语句每个变量是什么意思**，只有把工作做细到这个程度，才是真正对我们这些小白有利（大神们请忽略，这只是给小白们看的。）本篇是系列教程的第一篇，详细阐述程序darknet.py。

话不多说，先看darknet.py代码的超详细注释。

```python
from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
from util import * 


def get_test_input():
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (416,416))          #Resize to the input dimension
    img_ =  img[:,:,::-1].transpose((2,0,1))  #img是【h,w,channel】，这里的img[:,:,::-1]是将第三个维度channel从opencv的BGR转化为pytorch的RGB，然后transpose((2,0,1))的意思是将[height,width,channel]->[channel,height,width]
    img_ = img_[np.newaxis,:,:,:]/255.0       #Add a channel at 0 (for batch) | Normalise
    img_ = torch.from_numpy(img_).float()     #Convert to float
    img_ = Variable(img_)                     # Convert to Variable
    return img_

def parse_cfg(cfgfile):
    """
    输入: 配置文件路径
    返回值: 列表对象,其中每一个元素为一个字典类型对应于一个要建立的神经网络模块（层）
    
    """
    # 加载文件并过滤掉文本中多余内容
    file = open(cfgfile, 'r')
    lines = file.read().split('\n')                        # store the lines in a list等价于readlines
    lines = [x for x in lines if len(x) > 0]               # 去掉空行
    lines = [x for x in lines if x[0] != '#']              # 去掉以#开头的注释行
    lines = [x.rstrip().lstrip() for x in lines]           # 去掉左右两边的空格(rstricp是去掉右边的空格，lstrip是去掉左边的空格)
    # cfg文件中的每个块用[]括起来最后组成一个列表，一个block存储一个块的内容，即每个层用一个字典block存储。
    block = {}
    blocks = []
    
    for line in lines:
        if line[0] == "[":               # 这是cfg文件中一个层(块)的开始           
            if len(block) != 0:          # 如果块内已经存了信息, 说明是上一个块的信息还没有保存
                blocks.append(block)     # 那么这个块（字典）加入到blocks列表中去
                block = {}               # 覆盖掉已存储的block,新建一个空白块存储描述下一个块的信息(block是字典)
            block["type"] = line[1:-1].rstrip()  # 把cfg的[]中的块名作为键type的值   
        else:
            key,value = line.split("=") #按等号分割
            block[key.rstrip()] = value.lstrip()#左边是key(去掉右空格)，右边是value(去掉左空格)，形成一个block字典的键值对
    blocks.append(block) # 退出循环，将最后一个未加入的block加进去
    # print('\n\n'.join([repr(x) for x in blocks]))
    return blocks

# 配置文件定义了6种不同type
# 'net': 相当于超参数,网络全局配置的相关参数
# {'convolutional', 'net', 'route', 'shortcut', 'upsample', 'yolo'}

# cfg = parse_cfg("cfg/yolov3.cfg")
# print(cfg)



class EmptyLayer(nn.Module):
    """
    为shortcut layer / route layer 准备, 具体功能不在此实现，在Darknet类的forward函数中有体现
    """
    def __init__(self):
        super(EmptyLayer, self).__init__()
        

class DetectionLayer(nn.Module):
    '''yolo 检测层的具体实现, 在特征图上使用锚点预测目标区域和类别, 功能函数在predict_transform中'''
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors

def create_modules(blocks):
    net_info = blocks[0]     # blocks[0]存储了cfg中[net]的信息，它是一个字典，获取网络输入和预处理相关信息    
    module_list = nn.ModuleList() # module_list用于存储每个block,每个block对应cfg文件中一个块，类似[convolutional]里面就对应一个卷积块
    prev_filters = 3   #初始值对应于输入数据3通道，用来存储我们需要持续追踪被应用卷积层的卷积核数量（上一层的卷积核数量（或特征图深度））
    output_filters = []   #我们不仅需要追踪前一层的卷积核数量，还需要追踪之前每个层。随着不断地迭代，我们将每个模块的输出卷积核数量添加到 output_filters 列表上。
    
    for index, x in enumerate(blocks[1:]): #这里，我们迭代block[1:] 而不是blocks，因为blocks的第一个元素是一个net块，它不属于前向传播。
        module = nn.Sequential()# 这里每个块用nn.sequential()创建为了一个module,一个module有多个层
    
        #check the type of block
        #create a new module for the block
        #append to module_list
        
        if (x["type"] == "convolutional"):
            ''' 1. 卷积层 '''
            # 获取激活函数/批归一化/卷积层参数（通过字典的键获取值）
            activation = x["activation"]
            try:
                batch_normalize = int(x["batch_normalize"])
                bias = False#卷积层后接BN就不需要bias
            except:
                batch_normalize = 0
                bias = True #卷积层后无BN层就需要bias
        
            filters= int(x["filters"])
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])
        
            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0
        
            # 开始创建并添加相应层
            # Add the convolutional layer
            # nn.Conv2d(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True)
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias = bias)
            module.add_module("conv_{0}".format(index), conv)
        
            #Add the Batch Norm Layer
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index), bn)
        
            #Check the activation. 
            #It is either Linear or a Leaky ReLU for YOLO
            # 给定参数负轴系数0.1
            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace = True)
                module.add_module("leaky_{0}".format(index), activn)
                   
        elif (x["type"] == "upsample"):
            '''
            2. upsampling layer
            没有使用 Bilinear2dUpsampling
            实际使用的为最近邻插值
            '''
            stride = int(x["stride"])#这个stride在cfg中就是2，所以下面的scale_factor写2或者stride是等价的
            upsample = nn.Upsample(scale_factor = 2, mode = "nearest")
            module.add_module("upsample_{}".format(index), upsample)
                
        # route layer -> Empty layer
        # route层的作用：当layer取值为正时，输出这个正数对应的层的特征，如果layer取值为负数，输出route层向后退layer层对应层的特征
        elif (x["type"] == "route"):
            x["layers"] = x["layers"].split(',')
            #Start  of a route
            start = int(x["layers"][0])
            #end, if there exists one.
            try:
                end = int(x["layers"][1])
            except:
                end = 0
            #Positive anotation: 正值
            if start > 0: 
                start = start - index            
            if end > 0:# 若end>0，由于end= end - index，再执行index + end输出的还是第end层的特征
                end = end - index
            route = EmptyLayer()
            module.add_module("route_{0}".format(index), route)
            if end < 0: #若end<0，则end还是end，输出index+end(而end<0)故index向后退end层的特征。
                filters = output_filters[index + start] + output_filters[index + end]
            else: #如果没有第二个参数，end=0，则对应下面的公式，此时若start>0，由于start = start - index，再执行index + start输出的还是第start层的特征;若start<0，则start还是start，输出index+start(而start<0)故index向后退start层的特征。
                filters= output_filters[index + start]
    
        #shortcut corresponds to skip connection
        elif x["type"] == "shortcut":
            shortcut = EmptyLayer() #使用空的层，因为它还要执行一个非常简单的操作（加）。没必要更新 filters 变量,因为它只是将前一层的特征图添加到后面的层上而已。
            module.add_module("shortcut_{}".format(index), shortcut)
            
        #Yolo is the detection layer
        elif x["type"] == "yolo":
            mask = x["mask"].split(",")
            mask = [int(x) for x in mask]
    
            anchors = x["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors),2)]
            anchors = [anchors[i] for i in mask]
    
            detection = DetectionLayer(anchors)# 锚点,检测,位置回归,分类，这个类见predict_transform中
            module.add_module("Detection_{}".format(index), detection)
                              
        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)
        
    return (net_info, module_list)

class Darknet(nn.Module):
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile) #调用parse_cfg函数
        self.net_info, self.module_list = create_modules(self.blocks)#调用create_modules函数
        
    def forward(self, x, CUDA):
        modules = self.blocks[1:] # 除了net块之外的所有，forward这里用的是blocks列表中的各个block块字典
        outputs = {}   #We cache the outputs for the route layer
        
        write = 0#write表示我们是否遇到第一个检测。write=0，则收集器尚未初始化，write=1，则收集器已经初始化，我们只需要将检测图与收集器级联起来即可。
        for i, module in enumerate(modules):        
            module_type = (module["type"])
            
            if module_type == "convolutional" or module_type == "upsample":
                x = self.module_list[i](x)
    
            elif module_type == "route":
                layers = module["layers"]
                layers = [int(a) for a in layers]
    
                if (layers[0]) > 0:
                    layers[0] = layers[0] - i
                # 如果只有一层时。从前面的if (layers[0]) > 0:语句中可知，如果layer[0]>0，则输出的就是当前layer[0]这一层的特征,如果layer[0]<0，输出就是从route层(第i层)向后退layer[0]层那一层得到的特征 
                if len(layers) == 1:
                    x = outputs[i + (layers[0])]
                #第二个元素同理 
                else:
                    if (layers[1]) > 0:
                        layers[1] = layers[1] - i
    
                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    x = torch.cat((map1, map2), 1)#第二个参数设为 1,这是因为我们希望将特征图沿anchor数量的维度级联起来。
                
    
            elif  module_type == "shortcut":
                from_ = int(module["from"])
                x = outputs[i-1] + outputs[i+from_] # 求和运算，它只是将前一层的特征图添加到后面的层上而已
            
            elif module_type == 'yolo':        
                anchors = self.module_list[i][0].anchors
                #从net_info(实际就是blocks[0]，即[net])中get the input dimensions
                inp_dim = int (self.net_info["height"])
        
                #Get the number of classes
                num_classes = int (module["classes"])
        
                #Transform 
                x = x.data # 这里得到的是预测的yolo层feature map
                # 在util.py中的predict_transform()函数利用x(是传入yolo层的feature map)，得到每个格子所对应的anchor最终得到的目标
                # 坐标与宽高，以及出现目标的得分与每种类别的得分。经过predict_transform变换后的x的维度是(batch_size, grid_size*grid_size*num_anchors, 5+类别数量)
                x = predict_transform(x, inp_dim, anchors, num_classes, CUDA)
                 
                if not write:              #if no collector has been intialised. 因为一个空的tensor无法与一个有数据的tensor进行concatenate操作，
                    detections = x #所以detections的初始化在有预测值出来时才进行，
                    write = 1   #用write = 1标记，当后面的分数出来后，直接concatenate操作即可。
        
                else:  
                    '''
                    变换后x的维度是(batch_size, grid_size*grid_size*num_anchors, 5+类别数量)，这里是在维度1上进行concatenate，即按照
                    anchor数量的维度进行连接，对应教程part3中的Bounding Box attributes图的行进行连接。yolov3中有3个yolo层，所以
                    对于每个yolo层的输出先用predict_transform()变成每行为一个anchor对应的预测值的形式(不看batch_size这个维度，x剩下的
                    维度可以看成一个二维tensor)，这样3个yolo层的预测值按照每个方框对应的行的维度进行连接。得到了这张图处所有anchor的预测值，后面的NMS等操作可以一次完成
                    '''
                    detections = torch.cat((detections, x), 1)# 将在3个不同level的feature map上检测结果存储在 detections 里
        
            outputs[i] = x
        
        return detections
# blocks = parse_cfg('cfg/yolov3.cfg')
# x,y = create_modules(blocks)
# print(y)

    def load_weights(self, weightfile):
        #Open the weights file
        fp = open(weightfile, "rb")
    
        #The first 5 values are header information 
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number 
        # 4,5. Images seen by the network (during training)
        header = np.fromfile(fp, dtype = np.int32, count = 5)# 这里读取first 5 values权重
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]   
        
        weights = np.fromfile(fp, dtype = np.float32)#加载 np.ndarray 中的剩余权重，权重是以float32类型存储的
        
        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"] # blocks中的第一个元素是网络参数和图像的描述，所以从blocks[1]开始读入
    
            #If module_type is convolutional load weights
            #Otherwise ignore.
            
            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i+1]["batch_normalize"]) # 当有bn层时，"batch_normalize"对应值为1
                except:
                    batch_normalize = 0
            
                conv = model[0]
                
                
                if (batch_normalize):
                    bn = model[1]
        
                    #Get the number of weights of Batch Norm Layer
                    num_bn_biases = bn.bias.numel()
        
                    #Load the weights
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases
        
                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
        
                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
        
                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
        
                    #Cast the loaded weights into dims of model weights. 
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)
        
                    #Copy the data to model 将从weights文件中得到的权重bn_biases复制到model中(bn.bias.data)
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)
                
                else:#如果 batch_normalize 的检查结果不是 True，只需要加载卷积层的偏置项
                    #Number of biases
                    num_biases = conv.bias.numel()
                
                    #Load the weights
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases
                
                    #reshape the loaded weights according to the dims of the model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)
                
                    #Finally copy the data
                    conv.bias.data.copy_(conv_biases)
                    
                #Let us load the weights for the Convolutional layers
                num_weights = conv.weight.numel()
                
                #Do the same as above for weights
                conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
                ptr = ptr + num_weights
                
                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)
```

　　总的来说，darknet.py程序包含函数parse_cfg输入 配置文件路径返回一个列表,其中每一个元素为一个字典类型对应于一个要建立的神经网络模块（层），而函数create_modules用来创建网络层级，而Darknet类的forward函数就是实现网络前向传播函数了，还有个load_weights用来导入预训练的网络权重参数。当然，forward函数中需要产生需要的预测输出形式，因此需要变换输出即函数 predict_transform 在文件 util.py 中，我们在 Darknet 类别的 forward 中使用该函数时，将导入该函数。下一篇就要详细注释util.py 了。

# 第二部分：utils.py

本篇接着上一篇去解释util.py。这个程序包含了predict_transform函数（Darknet类中的forward函数要用到），write_results函数使我们的输出满足 objectness 分数阈值和非极大值抑制（NMS），以得到「真实」检测结果。还有prep_image和letterbox_image等图片预处理函数等（前者用来将numpy数组转换成PyTorch需要的的输入格式，后者用来将图片按照纵横比进行缩放，将空白部分用(128,128,128)填充）。话不多说，直接看util.py的注释。

```python
from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
import cv2 

def unique(tensor):#因为同一类别可能会有多个真实检测结果，所以我们使用unique函数来去除重复的元素，即一类只留下一个元素，达到获取任意给定图像中存在的类别的目的。
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)#np.unique该函数是去除数组中的重复数字，并进行排序之后输出
    unique_tensor = torch.from_numpy(unique_np)
    # 复制数据
    tensor_res = tensor.new(unique_tensor.shape)# new(args, *kwargs) 构建[相同数据类型]的新Tensor
    tensor_res.copy_(unique_tensor)
    return tensor_res


def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes 
    
    
    """
    #Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]
    
    #get the corrdinates of the intersection rectangle
    inter_rect_x1 =  torch.max(b1_x1, b2_x1)
    inter_rect_y1 =  torch.max(b1_y1, b2_y1)
    inter_rect_x2 =  torch.min(b1_x2, b2_x2)
    inter_rect_y2 =  torch.min(b1_y2, b2_y2)
    
    #Intersection area
    # Intersection area 这里没有对inter_area为负的情况进行判断，后面计算出来的IOU就可能是负的
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)

    #Union Area
    b1_area = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1)*(b2_y2 - b2_y1 + 1)
    
    iou = inter_area / (b1_area + b2_area - inter_area)
    
    return iou

def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA = True):
    """
    在特征图上进行多尺度预测, 在GRID每个位置都有三个不同尺度的锚点.predict_transform()利用一个scale得到的feature map预测得到的每个anchor的属性(x,y,w,h,s,s_cls1,s_cls2...),其中x,y,w,h
    是在网络输入图片坐标系下的值,s是方框含有目标的置信度得分，s_cls1,s_cls_2等是方框所含目标对应每类的概率。输入的feature map(prediction变量) 
    维度为(batch_size, num_anchors*bbox_attrs, grid_size, grid_size)，类似于一个batch彩色图片BxCxHxW存储方式。参数见predict_transform()里面的变量。
    并且将结果的维度变换成(batch_size, grid_size*grid_size*num_anchors, 5+类别数量)的tensor，同时得到每个方框在网络输入图片(416x416)坐标系下的(x,y,w,h)以及方框含有目标的得分以及每个类的得分。
    """
    batch_size = prediction.size(0)
    # stride表示的是整个网络的步长，等于图像原始尺寸与yolo层输入的feature mapr尺寸相除，因为输入图像是正方形，所以用高相除即可
    stride =  inp_dim // prediction.size(2)#416//13=32
    # feature map每条边格子的数量，416//32=13
    grid_size = inp_dim // stride
    # 一个方框属性个数，等于5+类别数量
    bbox_attrs = 5 + num_classes
    # anchors数量
    num_anchors = len(anchors)   
    # 输入的prediction维度为(batch_size, num_anchors * bbox_attrs, grid_size, grid_size)，类似于一个batch彩色图片BxCxHxW
    # 存储方式，将它的维度变换成(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    #contiguous：view只能用在contiguous的variable上。如果在view之前用了transpose, permute等，需要用contiguous()来返回一个contiguous copy。
    prediction = prediction.transpose(1,2).contiguous()
    # 将prediction维度转换成(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)。不看batch_size，
    # (grid_size*grid_size*num_anchors, bbox_attrs)相当于将所有anchor按行排列，即一行对应一个anchor属性，此时的属性仍然是feature map得到的值
    prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)
    # 锚点的维度与net块的height和width属性一致。这些属性描述了输入图像的维度，比feature map的规模大（二者之商即是步幅）。因此，我们必须使用stride分割锚点。变换后的anchors是相对于最终的feature map的尺寸
    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]

    #Sigmoid the tX, tY. and object confidencce.tx与ty为预测的坐标偏移值
    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])
    
    #这里生成了每个格子的左上角坐标，生成的坐标为grid x grid的二维数组，a，b分别对应这个二维矩阵的x,y坐标的数组，a,b的维度与grid维度一样。每个grid cell的尺寸均为1，故grid范围是[0,12]（假如当前的特征图13*13）
    grid = np.arange(grid_size)
    a,b = np.meshgrid(grid, grid)
    #x_offset即cx,y_offset即cy，表示当前cell左上角坐标
    x_offset = torch.FloatTensor(a).view(-1,1)#view是reshape功能，-1表示自适应
    y_offset = torch.FloatTensor(b).view(-1,1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()
    #这里的x_y_offset对应的是最终的feature map中每个格子的左上角坐标，比如有13个格子，刚x_y_offset的坐标就对应为(0,0),(0,1)…(12,12) .view(-1, 2)将tensor变成两列，unsqueeze(0)在0维上添加了一维。
    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)

    prediction[:,:,:2] += x_y_offset#bx=sigmoid(tx)+cx,by=sigmoid(ty)+cy

   
    anchors = torch.FloatTensor(anchors)

    if CUDA:
        anchors = anchors.cuda()
    # 这里的anchors本来是一个长度为6的list(三个anchors每个2个坐标)，然后在0维上(行)进行了grid_size*grid_size个复制，在1维(列)上
    # 一次复制(没有变化)，即对每个格子都得到三个anchor。Unsqueeze(0)的作用是在数组上添加一维，这里是在第0维上添加的。添加grid_size是为了之后的公式bw=pw×e^tw的tw。
    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    #对网络预测得到的矩形框的宽高的偏差值进行指数计算，然后乘以anchors里面对应的宽高(这里的anchors里面的宽高是对应最终的feature map尺寸grid_size)，
    # 得到目标的方框的宽高，这里得到的宽高是相对于在feature map的尺寸
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors#公式bw=pw×e^tw及bh=ph×e^th，pw为anchorbox的长度
    # 这里得到每个anchor中每个类别的得分。将网络预测的每个得分用sigmoid()函数计算得到
    prediction[:,:,5: 5 + num_classes] = torch.sigmoid((prediction[:,:, 5 : 5 + num_classes]))

    prediction[:,:,:4] *= stride#将相对于最终feature map的方框坐标和尺寸映射回输入网络图片(416x416)，即将方框的坐标乘以网络的stride即可
    
    return prediction



    '''
    必须使我们的输出满足 objectness 分数阈值和非极大值抑制（NMS），以得到后文所提到的「真实」检测结果。要做到这一点就要用 write_results函数。
    函数的输入为预测结果、置信度（objectness 分数阈值）、num_classes（我们这里是 80）和 nms_conf（NMS IoU 阈值）。
    write_results()首先将网络输出方框属性(x,y,w,h)转换为在网络输入图片(416x416)坐标系中，方框左上角与右下角坐标(x1,y1,x2,y2)，以方便NMS操作。
    然后将方框含有目标得分低于阈值的方框去掉，提取得分最高的那个类的得分max_conf，同时返回这个类对应的序号max_conf_score,
    然后进行NMS操作。最终每个方框的属性为(ind,x1,y1,x2,y2,s,s_cls,index_cls)，ind 是这个方框所属图片在这个batch中的序号，
    x1,y1是在网络输入图片(416x416)坐标系中，方框左上角的坐标；x2,y2是在网络输入图片(416x416)坐标系中，方框右下角的坐标。
    s是这个方框含有目标的得分,s_cls是这个方框中所含目标最有可能的类别的概率得分，index_cls是s_cls对应的这个类别所对应的序号.
    '''
def write_results(prediction, confidence, num_classes, nms_conf = 0.4):
    # confidence: 输入的预测shape=(1,10647, 85)。conf_mask: shape=(1,10647) => 增加一维度之后 (1, 10647, 1)
    conf_mask = (prediction[:,:,4] > confidence).float().unsqueeze(2)#我们的预测张量包含有关Bx10647边界框的信息。对于含有目标的得分小于confidence的每个方框，它对应的含有目标的得分将变成0,即conf_mask中对应元素为0.而保留预测结果中置信度大于给定阈值的部分prediction的conf_mask
    prediction = prediction*conf_mask # 小于置信度的条目值全为0, 剩下部分不变。conf_mask中含有目标的得分小于confidence的方框所对应的含有目标的得分为0，
    #根据numpy的广播原理，它会扩展成与prediction维度一样的tensor，所以含有目标的得分小于confidence的方框所有的属性都会变为0，故如果没有检测任何有效目标,则返回值为0
   
    '''
    保留预测结果中置信度大于阈值的bbox
    下面开始为nms准备
    '''

    # prediction的前五个数据分别表示 (Cx, Cy, w, h, score)，这里创建一个新的数组，大小与predicton的大小相同   
    box_corner = prediction.new(prediction.shape)
    '''
    我们可以将我们的框的 (中心 x, 中心 y, 高度, 宽度) 属性转换成 (左上角 x, 左上角 y, 右下角 x, 右下角 y)
    这样做用每个框的两个对角坐标能更轻松地计算两个框的 IoU
    '''
    box_corner[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)# x1 = Cx - w/2
    box_corner[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)# y1 = Cy - h/2
    box_corner[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2)# x2 = Cx + w/2 
    box_corner[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)# y2 = Cy + h/2
    prediction[:,:,:4] = box_corner[:,:,:4]# 计算后的新坐标复制回去
    
    batch_size = prediction.size(0)# 第0个维度是batch_size
    # output = prediction.new(1, prediction.size(2)+1) # shape=(1,85+1)
    write = False # 拼接结果到output中最后返回
    
    #对每一张图片得分的预测值进行NMS操作，因为每张图片的目标数量不一样，所以有效得分的方框的数量不一样，没法将几张图片同时处理，因此一次只能完成一张图的置信度阈值的设置和NMS,不能将所涉及的操作向量化.
    #所以必须在预测的第一个维度上（batch数量）上遍历每张图片，将得分低于一定分数的去掉，对剩下的方框进行进行NMS
    for ind in range(batch_size):
        image_pred = prediction[ind]          # 选择此batch中第ind个图像的预测结果,image_pred对应一张图片中所有方框的坐标(x1,y1,x2,y2)以及得分，是一个二维tensor 维度为10647x85
      
        # 最大值索引, 最大值, 按照dim=1 方向计算
        max_conf, max_conf_score = torch.max(image_pred[:,5:5+ num_classes], 1)#我们只关心有最大值的类别分数，prediction[:, 5:]表示每一分类的分数,返回每一行中所有类别的得分最高的那个类的得分max_conf，同时返回这个类对应的序号max_conf_score
        # 维度扩展max_conf: shape=(10647->15) => (10647->15,1)添加一个列的维度，max_conf变成二维tensor，尺寸为10647x1
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        seq = (image_pred[:,:5], max_conf, max_conf_score)#我们移除了每一行的这 80 个类别分数，只保留bbox4个坐标以及objectnness分数，转而增加了有最大值的类别分数及索引。
        #将每个方框的(x1,y1,x2,y2,s)与得分最高的这个类的分数s_cls(max_conf)和对应类的序号index_cls(max_conf_score)在列维度上连接起来，
        # 即将10647x5,10647x1,10647x1三个tensor 在列维度进行concatenate操作，得到一个10647x7的tensor,(x1,y1,x2,y2,s,s_cls,index_cls)。
        image_pred = torch.cat(seq, 1)# shape=(10647, 5+1+1=7)
        #image_pred[:,4]是长度为10647的一维tensor,维度为4的列是置信度分数。假设有15个框含有目标的得分非0，返回15x1的tensor
        non_zero_ind =  (torch.nonzero(image_pred[:,4]))#torch.nonzero返回的是索引，会让non_zero_ind是个2维tensor
        try:# try-except模块的目的是处理无检测结果的情况.non_zero_ind.squeeze()将15x1的non_zero_ind去掉维度为1的维度，变成长度为15的一维tensor，相当于一个列向量，
            # image_pred[non_zero_ind.squeeze(),:]是在image_pred中找到non_zero_ind中非0目标得分的行的所有元素(image_pred维度
            # 是10647x7，找到其中的15行)， 再用view(-1,7)将它变为15x7的tensor，用view()确保第二个维度必须是7.
            image_pred_ = image_pred[non_zero_ind.squeeze(),:].view(-1,7)
        except:
            continue
        
        if image_pred_.shape[0] == 0:#当没有检测到时目标时，我们使用 continue 来跳过对本图像的循环，即进行下一次循环。
            continue             
  
         # 获取当前图像检测结果中出现的所有类别
        img_classes = unique(image_pred_[:,-1])  #pred_[:,-1]是一个15x7的tensor，最后一列保存的是每个框里面物体的类别，-1表示取最后一列。
        #用unique()除去重复的元素，即一类只留下一个元素，假设这里最后只剩下了3个元素，即只有3类物体。
        
        #按照类别执行 NMS
        
        for cls in img_classes:
            #一旦我们进入循环，我们要做的第一件事就是提取特定类别（用变量 cls 表示）的检测结果,分离检测结果中属于当前类的数据 -1: index_cls, -2: s_cls
            '''
            本句是将image_pred_中属于cls类的预测值保持不变，其余的全部变成0。image_pred_[:,-1] == cls，返回一个与image_pred_
            行数一样的一维tensor，这里长度为15.当image_pred_中的最后一个元素(物体类别索引)等于第cls类时，返回的tensor对应元素为1，
            否则为0. 它与image_pred_相乘时，先扩展为15x7的tensor(似乎这里还没有变成15x7的tensor)，为0元素一行全部为0，再与
            image_pred_相乘，属于cls这类的方框对应预测元素不变，其它类的为0.unsqueeze(1)添加了列这一维，变成15x7的二维tensor。
            '''
            cls_mask = image_pred_*(image_pred_[:,-1] == cls).float().unsqueeze(1) 
            class_mask_ind = torch.nonzero(cls_mask[:,-2]).squeeze() #cls_mask[:,-2]为cls_mask倒数第二列,是物体类别分数。
#cls_mask本身为15x7，cls_mask[:,-2]将cls_mask的倒数第二列取出来，此时是1维tensor，torch.nonzero(cls_mask[:,-2])得到的是非零元素的索引，
#将返回一个二维tensor，这里是4x2，再用squeeze()去掉长度为1的维度(这里是第二维)，得到一维tensor(相当于一列)。
            image_pred_class = image_pred_[class_mask_ind].view(-1,7) #从prediction中取出属于cls类别的所有结果，为下一步的nms的输入.
#找到image_pred_中对应cls类的所有方框的预测值，并转换为二维张量。这里4x7。image_pred_[class_mask_ind]本身得到的数据就是4x7，view(-1,7)是为了确保第二维为7
            ''' 到此步 prediction_class 已经存在了我们需要进行非极大值抑制的数据 '''
            # 开始 nms
            # 按照score排序, 由大到小
            # 最大值最上面            
            
            conf_sort_index = torch.sort(image_pred_class[:,4], descending = True )[1]# # 这里的sort()将返回两个tensor，第一个是每个框含有有目标的分数由低到高排列，第二个是现在由高到底的tensor中每个元素在原来的序号。[0]是排序结果, [1]是排序结果的索引
            image_pred_class = image_pred_class[conf_sort_index]#根据排序后的索引对应出的bbox的坐标与分数，依然为4x7的tensor
            idx = image_pred_class.size(0)   #detections的个数

            '''开始执行 "非极大值抑制" 操作'''
            for i in range(idx):
                # 对已经有序的结果，每次开始更新后索引加一，挨个与后面的结果比较
                
                try:# image_pred_class[i].unsqueeze(0)，为什么要加unsqueeze(0)？这里image_pred_class为4x7的tensor，image_pred_class[i]是一个长度为7的tensor，要变成1x7的tensor，在第0维添加一个维度。
                    ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i+1:])#这句话的作用是计算第i个方框和i+1到最终的所有方框的IOU。
                except ValueError:
                '''
                在for i in range(idx):这个循环中，因为有一些框(在image_pred_class对应一行)会被去掉，image_pred_class行数会减少，
                这样在后面的循环中，idx序号会超出image_pred_class的行数的范围，出现ValueError错误。
                所以当抛出这个错误时，则跳出这个循环，因为此时已经没有更多可以去掉的方框了。
                '''
                    break
            
                except IndexError:
                    break
            
                
                iou_mask = (ious < nms_conf).float().unsqueeze(1)# 计算出需要保留的item（保留ious < nms_conf的框）而ious < nms_conf得到的是torch.uint8类型，用float()将它们转换为float类型。因为要与image_pred_class[i+1:]相乘，故长度为7的tensor，要变成1x7的tensor，需添加一个维度。
                image_pred_class[i+1:] *= iou_mask #将iou_mask与比序号i大的框的预测值相乘，其中IOU大于阈值的框的预测值全部变成0.得出需要保留的框    
            
                # 开始移除
                non_zero_ind = torch.nonzero(image_pred_class[:,4]).squeeze()#torch.nonzero返回的是索引，是2维tensor。将经过iou_mask掩码后的每个方框含有目标的得分为非0的方框的索引提取出来，non_zero_ind经squeeze后为一维tensor，含有目标的得分非0的索引
                image_pred_class = image_pred_class[non_zero_ind].view(-1,7)#得到含有目标的得分非0的方框的预测值(x1, y1, x2, y2, s,  s_class,index_cls)，为1x7的tensor
            # 当前类的nms执行完之后，下一次循环将对剩下的方框中得分第i+1高的方框进行NMS操作，因为刚刚已经对得分第1到i高的方框进行了NMS操作。直到最后一个方框循环完成为止
            # 在每次进行NMS操作的时候，预测值tensor中都会有一些行(对应某些方框)被去掉。接下来是保存结果。
            # new()创建了一个和image_pred_class类型相同的tensor，tensor行数等于cls这个类别所有的方框经过NMS剩下的方框的个数，即image_pred_class的行数，列数为1.
            #再将生成的这个tensor所有元素赋值为这些方框所属图片对应于batch中的序号ind(一个batch有多张图片同时测试)，用fill_(ind)实现   
            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)     
            seq = batch_ind, image_pred_class
            #我们没有初始化我们的输出张量，除非我们有要分配给它的检测结果。一旦其被初始化，我们就将后续的检测结果与它连接起来。我们使用write标签来表示张量是否初始化了。在类别上迭代的循环结束时，我们将所得到的检测结果加入到张量输出中。
            if not write:
            # 将batch_ind, image_pred_class在列维度上进行连接，image_pred_class每一行存储的是(x1,y1,x2,y2,s,s_cls,index_cls)，现在在第一列增加了一个代表这个行对应方框所属图片在一个batch中的序号ind
                output = torch.cat(seq,1)
                write = True
            else:
                out = torch.cat(seq,1)
                output = torch.cat((output,out))
    
    try:#在该函数结束时，我们会检查输出是否已被初始化。如果没有，就意味着在该 batch 的任意图像中都没有单个检测结果。在这种情况下，我们返回 0。
        return output
    except:# 如果所有的图片都没有检测到方框，则在前面不会进行NMS等操作，不会生成output，此时将在except中返回0
        return 0
    # 最终返回的output是一个batch中所有图片中剩下的方框的属性，一行对应一个方框，属性为(x1,y1,x2,y2,s,s_cls,index_cls)，
    # ind 是这个方框所属图片在这个batch中的序号，x1,y1是在网络输入图片(416x416)坐标系中，方框左上角的坐标；x2,y2是在网络输入
    # 图片(416x416)坐标系中，方框右下角的坐标。s是这个方框含有目标的得分s_cls是这个方框中所含目标最有可能的类别的概率得分，index_cls是s_cls对应的这个类别所对应的序号

def letterbox_image(img, inp_dim):
    """
    lteerbox_image()将图片按照纵横比进行缩放，将空白部分用(128,128,128)填充,调整图像尺寸
    具体而言,此时某个边正好可以等于目标长度,另一边小于等于目标长度
    将缩放后的数据拷贝到画布中心,返回完成缩放
    """
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim　#inp_dim是需要resize的尺寸（如416*416）
    # 取min(w/img_w, h/img_h)这个比例来缩放，缩放后的尺寸为new_w, new_h,即保证较长的边缩放后正好等于目标长度(需要的尺寸)，另一边的尺寸缩放后还没有填充满.
    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))
    resized_image = cv2.resize(img, (new_w,new_h), interpolation = cv2.INTER_CUBIC) #将图片按照纵横比不变来缩放为new_w x new_h，768 x 576的图片缩放成416x312.,用了双三次插值
    # 创建一个画布, 将resized_image数据拷贝到画布中心。
    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)#生成一个我们最终需要的图片尺寸hxwx3的array,这里生成416x416x3的array,每个元素值为128
    # 将wxhx3的array中对应new_wxnew_hx3的部分(这两个部分的中心应该对齐)赋值为刚刚由原图缩放得到的数组,得到最终缩放后图片
    canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w,  :] = resized_image
    
    return canvas

def prep_image(img, inp_dim):#prep_image用来将numpy数组转换成PyTorch需要的的输入格式。即（3，416,416）
    """
    为神经网络准备输入图像数据
    返回值: 处理后图像, 原图, 原图尺寸
    """
    img = (letterbox_image(img, (inp_dim, inp_dim)))# lteerbox_image()将图片按照纵横比进行缩放，将空白部分用(128,128,128)填充
    img = img[:,:,::-1].transpose((2,0,1)).copy()#img是【h,w,channel】，这里的img[:,:,::-1]是将第三个维度channel从opencv的BGR转化为pytorch的RGB，然后transpose((2,0,1))的意思是将[height,width,channel]->[channel,height,width]
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)# from_numpy(()将ndarray数据转换为tensor格式，div(255.0)将每个元素除以255.0，进行归一化，unsqueeze(0)在0维上添加了一维，
# 从3x416x416变成1x3x416x416，多出来的一维表示batch。这里就将图片变成了BxCxHxW的pytorch格式
    return img

def load_classes(namesfile): #load_classes会返回一个字典——将每个类别的索引映射到其名称的字符串
    """
    加载类名文件
    :param namesfile:
    :return: 元组,包括类名数组和总类的个数
    """
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    return names
```

# 第三部分：detect.py

本篇是第三篇，主要是对detect.py的注释。在这一部分，我们将为我们的检测器构建输入和输出流程。这涉及到从磁盘读取图像，做出预测，使用预测结果在图像上绘制边界框，然后将它们保存到磁盘上。我们将引入一些命令行标签，以便能使用该网络的各种超参数进行一些实验。注意代码中有一处错误我进行了修改。源代码在计算scaling_factor时，用的scaling_factor = torch.min(416/im_dim_list,1)[0].view(-1,1)显然不对，应该使用用户输入的args.reso即改为scaling_factor = torch.min(int(args.reso)/im_dim_list,1)[0].view(-1,1)

接下来就开始吧。

```python
from __future__ import division
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
from util import *
import argparse
import os 
import os.path as osp
from darknet import Darknet
import pickle as pkl
import pandas as pd
import random

def arg_parse():
    """
    检测模块的参数转换
    
    """
    #创建一个ArgumentParser对象，格式: 参数名, 目标参数(dest是字典的key),帮助信息,默认值,类型 
    parser = argparse.ArgumentParser(description='YOLO v3 检测模型')
    
    parser.add_argument("--images", dest = 'images', help = 
                        "待检测图像目录",
                        default = "imgs", type = str)  # images是所有测试图片所在的文件夹
    parser.add_argument("--det", dest = 'det', help =   #det保存检测结果的目录
                        "检测结果保存目录",
                        default = "det", type = str)
    parser.add_argument("--bs", dest = "bs", help = "Batch size，默认为 1", default = 1)
    parser.add_argument("--confidence", dest = "confidence", help = "目标检测结果置信度阈值", default = 0.5)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS非极大值抑制阈值", default = 0.4)
    parser.add_argument("--cfg", dest = 'cfgfile', help = 
                        "配置文件",
                        default = "cfg/yolov3.cfg", type = str)
    parser.add_argument("--weights", dest = 'weightsfile', help = 
                        "模型权重",
                        default = "yolov3.weights", type = str)
    parser.add_argument("--reso", dest = 'reso', help = 
                        "网络输入分辨率. 分辨率越高,则准确率越高; 反之亦然.",
                        default = "416", type = str)#reso输入图像的分辨率，可用于在速度与准确度之间的权衡
    parser.add_argument("--scales", dest="scales", help="缩放尺度用于检测", default="1,2,3", type=str)
    return parser.parse_args()# 返回转换好的结果
    
args = arg_parse()# args是一个namespace类型的变量，即argparse.Namespace, 可以像easydict一样使用,就像一个字典，key来索引变量的值   
# Namespace(bs=1, cfgfile='cfg/yolov3.cfg', confidence=0.5,det='det', images='imgs', nms_thresh=0.4, reso='416', weightsfile='yolov3.weights')
images = args.images
batch_size = int(args.bs)
confidence = float(args.confidence)
nms_thesh = float(args.nms_thresh)
start = 0
CUDA = torch.cuda.is_available()# GPU环境是否可用



num_classes = 80# coco 数据集有80类
classes = load_classes("data/coco.names") #将类别文件载入到我们的程序中，coco.names文件中保存的是所有类别的名字，load_classes()返回一个列表classes，每个元素是一个类别的名字



#初始化网络并载入权重
print("载入神经网络...") 
model = Darknet(args.cfgfile)# Darknet类中初始化时得到了网络结构和网络的参数信息，保存在net_info，module_list中
model.load_weights(args.weightsfile)# 将权重文件载入，并复制给对应的网络结构model中
print("模型加载成功.")
# 网络输入数据大小
model.net_info["height"] = args.reso # model类中net_info是一个字典。’’height’’是图片的宽高，因为图片缩放到416x416，所以宽高一样大
inp_dim = int(model.net_info["height"])　#inp_dim是网络输入图片尺寸（如416*416）
assert inp_dim % 32 == 0 # 如果设定的输入图片的尺寸不是32的位数或者不大于32，抛出异常
assert inp_dim > 32

 # 如果GPU可用, 模型切换到cuda中运行
if CUDA:
    model.cuda()



model.eval()#变成测试模式，这主要是对dropout和batch normalization的操作在训练和测试的时候是不一样的

read_dir = time.time() #read_dir 是一个用于测量时间的检查点,开始计时
# 加载待检测图像列表
try: #从磁盘读取图像或从目录读取多张图像。图像的路径存储在一个名为 imlist 的列表中,imlist列表保存了images文件中所有图片的完整路径，一张图片路径对应一个元素。 
     #osp.realpath('.')得到了图片所在文件夹的绝对路径，images是测试图片文件夹，listdir(images)得到了images文件夹下面所有图片的名字。
     #通过join()把目录（文件夹）的绝对路径和图片名结合起来，就得到了一张图片的完整路径
    imlist = [osp.join(osp.realpath('.'), images, img) for img in os.listdir(images)]
except NotADirectoryError:# 如果上面的路径有错，只得到images文件夹绝对路径即可
    imlist = []
    imlist.append(osp.join(osp.realpath('.'), images))
except FileNotFoundError:
    print ("No file or directory with the name {}".format(images))
    exit()
# 存储结果目录    
if not os.path.exists(args.det): #如果保存检测结果的目录（由 det 标签定义）不存在，就创建一个
    os.makedirs(args.det)

load_batch = time.time()# 开始载入图片的时间。 load_batch - read_dir 得到读取所有图片路径的时间
loaded_ims = [cv2.imread(x) for x in imlist] #使用 OpenCV 来加载图像，将所有的图片读入，一张图片的数组在loaded_ims列表中保存为一个元素

# 加载全部待检测图像
# loaded_ims和[inp_dim for x in range(len(imlist))]是两个列表，lodded_ims是所有图片数组的列表，[inp_dim for x in range(len(imlist))] 遍历imlist长度(即图片的数量)这么多次，每次返回值是图片需resize的输入尺寸inp_dim（如416）
im_batches = list(map(prep_image, loaded_ims, [inp_dim for x in range(len(imlist))]))#map函数将对应的元素作为参数传入prep_image函数，最终的所有结果也会组成一个列表(im_batches)，是BxCxHxW
im_dim_list = [(x.shape[1], x.shape[0]) for x in loaded_ims]#除了转换后的图像，我们也会维护一个列表im_dim_list用于保存原始图片的维度。一个元素对应一张图片的宽高,opencv读入的图片矩阵对应的是 HxWxC
#将im_dim_list转换为floatTensor类型的tensor，此时维度为11x2，（因为本例测试集一共11张图片）并且每个元素沿着第二维(列的方向)进行复制，最终变成11x4的tensor。一行的元素为(W,H,W,H)，对应一张图片原始的宽、高，且重复了一次。(W,H,W,H)主要是在后面计算x1,y1,x2,y2各自对应的缩放系数时好对应上。
im_dim_list = torch.FloatTensor(im_dim_list).repeat(1,2)#repeat(*size), 沿着指定维度复制数据，size维度必须和数据本身维度要一致

leftover = 0 #创建 batch，将所有测试图片按照batch_size分成多个batch
if (len(im_dim_list) % batch_size):# 如果测试图片的数量不能被batch_size整除，leftover=1
    leftover = 1
#如果batch size 不等于1，则将一个batch的图片作为一个元素保存在im_batches中，按照if语句里面的公式计算。如果batch_size=1,则每一张图片作为一个元素保存在im_batches中
if batch_size != 1:
# 如果batch_size 不等于1，则batch的数量=图片数量//batch_size + leftover(测试图片的数量不能被batch_size整除，leftover=1，否则为0)。本例有11张图片，假设batch_size=2,则batch数量=6
    num_batches = len(imlist) // batch_size + leftover
# 前面的im_batches变量将所有的图片以BxCxHxW的格式保存。而这里将一个batch的所有图片在B这个维度(第0维度)上进行连接，torch.cat()默认在0维上进行连接。将这个连接后的tensor作为im_batches列表的一个元素。
#第i个batch在前面的im_batches变量中所对应的元素就是i*batch_size: (i + 1)*batch_size，但是最后一个batch如果用(i + 1)*batch_size可能会超过图片数量的len(im_batches)长度，所以取min((i + 1)*batch_size, len(im_batches)            
    im_batches = [torch.cat((im_batches[i*batch_size : min((i +  1)*batch_size,
                        len(im_batches))]))  for i in range(num_batches)]  

# The Detection Loop
write = 0


if CUDA:
    im_dim_list = im_dim_list.cuda()
# 开始计时，计算开始检测的时间。start_det_loop - load_batch 为读入所有图片并将它们分成不同batch的时间    
start_det_loop = time.time()
# enumerate返回im_batches列表中每个batch在0维连接成一个元素的tensor和这个tensor在im_batches中的序号。本例子中batch只有一张图片
for i, batch in enumerate(im_batches):
#load the image 
    start = time.time()
    if CUDA:
        batch = batch.cuda()
    # 取消梯度计算
    with torch.no_grad():
    # Variable(batch)将图片生成一个可导tensor，现在已经不再支持这种写法，Autograd automatically supports Tensors with requires_grad set to True。
    # prediction是一个batch所有图片通过yolov3模型得到的预测值，维度为1x10647x85，三个scale的图片每个scale的特征图大小为13x13,26x26,52x52,一个元素看作一个格子，每个格子有3个anchor，将一个anchor保存为一行，
    #所以prediction一共有(13x13+26x26+52x52)x3=10647行，一个anchor预测(x,y,w,h,s,s_cls1,s_cls2...s_cls_80)，一共有85个元素。所以prediction的维度为Bx10647x85，加为这里batch_size为1，所以prediction的维度为1x10647x85
        prediction = model(Variable(batch), CUDA)
    # 结果过滤.这里返回了经过NMS后剩下的方框，最终每个方框的属性为(ind,x1,y1,x2,y2,s,s_cls,index_cls) ind是这个方框所属图片在这个batch中的序号，x1,y1是在网络输入图片(416x416)坐标系中，方框左上角的坐标；x2,y2是方框右下角的坐标。
    # s是这个方框含有目标的得分，s_cls是这个方框中所含目标最有可能的类别的概率得分，index_cls是s_cls对应的这个类别在所有类别中所对应的序号。这里prediction维度是3x8，表示有3个框
    prediction = write_results(prediction, confidence, num_classes, nms_conf = nms_thesh)

    end = time.time()
    # 如果从write_results()返回的一个batch的结果是一个int(0)，表示没有检测到时目标，此时用continue跳过本次循环
    if type(prediction) == int:
    # 在imlist中，遍历一个batch所有的图片对应的元素(即每张图片的存储位置和名字)，同时返回这张图片在这个batch中的序号im_num
        for im_num, image in enumerate(imlist[i*batch_size: min((i +  1)*batch_size, len(imlist))]):
　　　　　　　　　　　　# 计算图片在imlist中所对应的序号,即在所有图片中的序号
            im_id = i*batch_size + im_num
　　　　　　　　　　　　# 打印图片运行的时间，用一个batch的平均运行时间来表示。.3f就表示保留三位小数点的浮点
            print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start)/batch_size))
　　　　　　　　　　　　# 输出本次处理图片所有检测到的目标的名字
            print("{0:20s} {1:s}".format("Objects Detected:", ""))
            print("----------------------------------------------------------")
        continue
　　　　# prediction[:,0]取出了每个方框在所在图片在这个batch(第i个batch)中的序号，加上i*batch_size，就将prediction中每个框(一行)的第一个元素（维度0）变成了这个框所在图片在imlist中的序号，即在所有图片中的序号
    prediction[:,0] += i*batch_size    
    # 这里用一个write标志来标记是否是第一次得到输出结果，因为每次的结果要进行torch.cat()操作，而一个空的变量不能与tensor连接，所以第一次将它赋值给output，后面就直接进行cat()操作
    if not write:                      #If we have't initialised output
        output = prediction  
        write = 1
    else:
    # output将每个batch的输出结果在0维进行连接，即在行维度上连接，每行表示一个检测方框的预测值。最终本例子中的11张图片检测得到的结果output维度为 34 x 8
        output = torch.cat((output,prediction))
    # 在imlist中，遍历一个batch所有的图片对应的元素(即每张图片的存储位置加名字)，同时返回这张图片在这个batch中的序号im_num
    for im_num, image in enumerate(imlist[i*batch_size: min((i +  1)*batch_size, len(imlist))]):
        im_id = i*batch_size + im_num# 计算图片在imlist中所对应的序号,即在所有图片中的序号
        # objs列表包含了本次处理图片中所有检测得到的方框所包含目标的类别名称。每个元素对应一个检测得到的方框所包含目标的类别名称。for x in output遍历output中的每一行(即一个方框的预测值)得到x，如果这个方
     　　 #框所在图片在所有图片中的序号等于本次处理图片的序号，则用classes[int(x[-1])找到这个方框包含目标类别在classes中对应的类的名字。
        objs = [classes[int(x[-1])] for x in output if int(x[0]) == im_id]# classes在之前的语句classes = load_classes("data/coco.names")中就是为了把类的序号转为字符名字
        print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start)/batch_size))# 打印本次处理图片运行的时间，用一个batch的平均运行时间来表示。.3f就表示保留三位小数点的浮点
        print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))# 输出本次处理图片所有检测到的目标的类别名字
        print("----------------------------------------------------------")

    if CUDA:
        torch.cuda.synchronize()  # 保证gpu和cpu同步，否则，一旦 GPU 工作排队了并且 GPU 工作还远未完成，那么 CUDA 核就将控制返回给 CPU（异步调用）。
# 对所有的输入的检测结果        
try:
#  check whether there has been a single detection has been made or not
    output
except NameError:
    print ("没有检测到任何目标")
    exit() # 当所有图片都有没检测到目标时，退出程序
# 最后输出output_recast - start_det_loop计算的是从开始检测，到去掉低分，NMS操作的时间.    
output_recast = time.time()
# 前面im_dim_list是一个4维tensor，一行的元素为(W,H,W,H)，对应一张图片原始的宽、高，且重复了一次。(W,H,W,H)主要是在后面计算x1,y1,x2,y2各自对应的缩放系数时好对应上。
#本例中im_dim_list维度为11x4.index_select()就是在im_dim_list中查找output中每行所对应方框所在图片在所有图片中的序号对应im_dim_list中的那一行，最终得到的im_dim_list的行数应该与output的行数相同。
#因此这样做后本例中此时im_dim_list维度34x4
im_dim_list = torch.index_select(im_dim_list, 0, output[:,0].long())# pytorch 切片torch.index_select(data, dim, indices)
"""
应该将方框的坐标转换为相对于填充后的图片中包含原始图片区域的计算方式。min(416/im_dim_list, 1)，416除以im_dim_list中的每个元素，然后在得到的tensor中的第1维(每行)去找到最小的元素.torch.min()返回一个
有两个tensor元素的tuple，第一个元素就是找到最小的元素的结果，这里没有给定 keepdim=True的标记，所以得到的最小元素的tensor会比原来减小一维，
另一个是每个最小值在每行中对应的序号。torch.min(416/im_dim_list, 1)[0]得到长度为34的最小元素构成的tensor，通过view(-1, 1)
变成了维度为34x1的tensor。这个tensor，即scaling_factor的每个元素就对应一张图片缩放成416的时候所采用的缩放系数
注意了！！！ Scaling_factor在进行计算的时候用的416，如果是其它的尺寸，这里不应该固定为416，在开始检测时util.py里所用的缩放系数就是用的 min(w/img_w, h/img_h)    
    
"""

#scaling_factor = torch.min(416/im_dim_list,1)[0].view(-1,1)#这是源代码，下面是我修改的代码
scaling_factor = torch.min(int(args.reso)/im_dim_list,1)[0].view(-1,1)
# 将相对于输入网络图片(416x416)的方框属性变换成原图按照纵横比不变进行缩放后的区域的坐标。
#scaling_factor*img_w和scaling_factor*img_h是图片按照纵横比不变进行缩放后的图片，即原图是768x576按照纵横比长边不变缩放到了416*372。
#经坐标换算,得到的坐标还是在输入网络的图片(416x416)坐标系下的绝对坐标，但是此时已经是相对于416*372这个区域的坐标了，而不再相对于(0,0)原点。
output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim_list[:,0].view(-1,1))/2#x1=x1−(416−scaling_factor*img_w)/2,x2=x2-(416−scaling_factor*img_w)/2
output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim_list[:,1].view(-1,1))/2#y1=y1-(416−scaling_factor*img_h)/2,y2=y2-(416−scaling_factor*img_h)/2


# 将方框坐标(x1,y1,x2,y2)映射到原始图片尺寸上，直接除以缩放系数即可。output[:,1:5]维度为34x4，scaling_factor维度是34x1.相除时会利用广播性质将scaling_factor扩展为34x4的tensor
output[:,1:5] /= scaling_factor # 缩放至原图大小尺寸

# 如果映射回原始图片中的坐标超过了原始图片的区域，则x1,x2限定在[0,img_w]内，img_w为原始图片的宽度。如果x1,x2小于0.0，令x1,x2为0.0，如果x1,x2大于原始图片宽度，令x1,x2大小为图片的宽度。
#同理，y1,y2限定在0,img_h]内，img_h为原始图片的高度。clamp()函数就是将第一个输入对数的值限定在后面两个数字的区间
for i in range(output.shape[0]):
    output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim_list[i,0])
    output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim_list[i,1])
    

class_load = time.time()# 开始载入颜色文件的时间
 # 绘图
colors = pkl.load(open("pallete", "rb"))# 读入包含100个颜色的文件pallete，里面是100个三元组序列

draw = time.time() # 开始画方框的文字的时间

# x为映射到原始图片中一个方框的属性(ind,x1,y1,x2,y2,s,s_cls,index_cls)，results列表保存了所有测试图片，一个元素对应一张图片
def write(x, results):
   
    c1 = tuple(x[1:3].int())# c1为方框左上角坐标x1,y1
    c2 = tuple(x[3:5].int()) # c2为方框右下角坐标x2,y2
    img = results[int(x[0])]# 在results中找到x方框所对应的图片，x[0]为方框所在图片在所有测试图片中的序号
    cls = int(x[-1])
    color = random.choice(colors)  # 随机选择一个颜色，用于后面画方框的颜色
    label = "{0}".format(classes[cls])# label为这个框所含目标类别名字的字符串
    cv2.rectangle(img, c1, c2,color, 1)# 在图片上画出(x1,y1,x2,y2)矩形，即我们检测到的目标方框
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0] # 得到一个包含目标名字字符的方框的宽高
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4　 # 得到包含目标名字的方框右下角坐标c2，这里在x,y方向上分别加了3、4个像素
    cv2.rectangle(img, c1, c2,color, -1) # 在图片上画一个实心方框，我们将在方框内放置目标类别名字
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1); # 在图片上写文字，(c1[0], c1[1] + t_size[1] + 4)为字符串的左下角坐标
    return img

# 开始逐条绘制output中结果.将每个框在对应图片上画出来，同时写上方框所含目标名字。map函数将output传递给map()中参数是函数的那个参数，每次传递一行。
#而lambda中x就是output中的一行，维度为1x8。loaded_ims列表保存了所有图片内容数组,一个元素对应一张图片，原地修改了loaded_ims 之中的图像，使之还包含了目标类别名字。
list(map(lambda x: write(x, loaded_ims), output))
#将带有方框的每张测试图片重新命名。det_names 是一个series对象，类似于一个列表，pd.Series(imlist)返回一个series对象。
#对于imlist这个列表(保存的是所有测试图片的绝对路径+名字，一个元素对应一张图片路径加名字)，生成的series对象包含两列，一列是每个imlist元素的索引，一列是 imlist 元素。
#apply()函数将这个series对象传递给apply()里面的函数，以遍历的方式进行。apply()返回结果是经过 apply()里面的函数返回每张测试图片将要保存的文件路径，这里依然是一个series对象
#x是Series()返回的对象中的一个元素，即一张图片的绝对路径加名字，args.det是将要保存图片的文件夹(默认det)，返回”det/det_图片名”,x.split("/")[-1]中的 ”/” 是linux下文件路径分隔符
det_names = pd.Series(imlist).apply(lambda x: "{}/det_{}".format(args.det,x.split("/")[-1]))# 每张图像都以「det_」加上图像名称的方式保存。我们创建了一个地址列表，这是我们保存我们的检测结果图像的位置。

list(map(cv2.imwrite, det_names, loaded_ims))# 保存标注了方框和目标类别名字的图片。det_names对应所有测试图片的保存路径，loaded_ims对应所有标注了方框和目标名字的图片数组


end = time.time()

print("SUMMARY")
print("----------------------------------------------------------")
print("{:25s}: {}".format("Task", "Time Taken (in seconds)"))
print()
print("{:25s}: {:2.3f}".format("Reading addresses", load_batch - read_dir))# 读取所有图片路径的时间
print("{:25s}: {:2.3f}".format("Loading batch", start_det_loop - load_batch))# 读入所有图片，并将图片按照batch size分成不同batch的时间
# 从开始检测到到去掉低分，NMS操作得到output的时间.  
print("{:25s}: {:2.3f}".format("Detection (" + str(len(imlist)) +  " images)", output_recast - start_det_loop))
#这里output映射回原图的时间
print("{:25s}: {:2.3f}".format("Output Processing", class_load - output_recast))
print("{:25s}: {:2.3f}".format("Drawing Boxes", end - draw))# 画框和文字的时间
print("{:25s}: {:2.3f}".format("Average time_per_img", (end - load_batch)/len(imlist)))# 从开始载入图片到所有结果处理完成，平均每张图片所消耗时间
print("----------------------------------------------------------")


torch.cuda.empty_cache()
```

# 第四部分：video.py

本篇介绍如何让检测器在视频或者网络摄像头上实时工作。我们将引入一些命令行标签，以便能使用该网络的各种超参数进行一些实验。这个代码是video.py，代码整体上很像detect.py，只有几处变化，只是我们不会在 batch 上迭代，而是在视频的帧上迭代。

注意代码中有一处错误我进行了修改。源代码在计算scaling_factor时，用的scaling_factor =  torch.min(416/im_dim,1)[0].view(-1,1)显然不对，应该使用用户输入的args.reso即改为scaling_factor  = torch.min(int(args.reso)/im_dim,1)[0].view(-1,1)

接下来就开始吧。

```python
from __future__ import division
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
from util import *
import argparse
import os 
import os.path as osp
from darknet import Darknet
import pickle as pkl
import pandas as pd
import random

def arg_parse():
    """
    视频检测模块的参数转换
    
    """
    #创建一个ArgumentParser对象，格式: 参数名, 目标参数(dest是字典的key),帮助信息,默认值,类型
    parser = argparse.ArgumentParser(description='YOLO v3 检测模型')
    parser.add_argument("--bs", dest = "bs", help = "Batch size，默认为 1", default = 1)
    parser.add_argument("--confidence", dest = "confidence", help = "目标检测结果置信度阈值", default = 0.5)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS非极大值抑制阈值", default = 0.4)
    parser.add_argument("--cfg", dest = 'cfgfile', help = 
                        "配置文件",
                        default = "cfg/yolov3.cfg", type = str)
    parser.add_argument("--weights", dest = 'weightsfile', help = 
                        "模型权重",
                        default = "yolov3.weights", type = str)
    parser.add_argument("--reso", dest = 'reso', help = 
                        "网络输入分辨率. 分辨率越高,则准确率越高; 反之亦然",
                        default = "416", type = str)
    parser.add_argument("--video", dest = "videofile", help = "待检测视频目录", default = "video.avi", type = str)
    
    return parser.parse_args()
    
args = arg_parse()# args是一个namespace类型的变量，即argparse.Namespace, 可以像easydict一样使用,就像一个字典，key来索引变量的值   
# Namespace(bs=1, cfgfile='cfg/yolov3.cfg', confidence=0.5,det='det', images='imgs', nms_thresh=0.4, reso='416', weightsfile='yolov3.weights')
batch_size = int(args.bs)
confidence = float(args.confidence)
nms_thesh = float(args.nms_thresh)
start = 0
CUDA = torch.cuda.is_available()# GPU环境是否可用



num_classes = 80# coco 数据集有80类
classes = load_classes("data/coco.names")#将类别文件载入到我们的程序中，coco.names文件中保存的是所有类别的名字，load_classes()返回一个列表classes，每个元素是一个类别的名字



#初始化网络并载入权重
print("载入神经网络....")
model = Darknet(args.cfgfile)# Darknet类中初始化时得到了网络结构和网络的参数信息，保存在net_info，module_list中
model.load_weights(args.weightsfile)# 将权重文件载入，并复制给对应的网络结构model中
print("模型加载成功.")
# 网络输入数据大小
model.net_info["height"] = args.reso# model类中net_info是一个字典。’’height’’是图片的宽高，因为图片缩放到416x416，所以宽高一样大
inp_dim = int(model.net_info["height"])#inp_dim是网络输入图片尺寸（如416*416）
assert inp_dim % 32 == 0 # 如果设定的输入图片的尺寸不是32的位数或者不大于32，抛出异常
assert inp_dim > 32

# 如果GPU可用, 模型切换到cuda中运行
if CUDA:
    model.cuda()


#变成测试模式，这主要是对dropout和batch normalization的操作在训练和测试的时候是不一样的
model.eval()


#要在视频或网络摄像头上运行这个检测器，代码基本可以保持不变，只是我们不会在 batch 上迭代，而是在视频的帧上迭代。
# 将方框和文字写在图片上
def write(x, results):
    c1 = tuple(x[1:3].int())# c1为方框左上角坐标x1,y1
    c2 = tuple(x[3:5].int())# c2为方框右下角坐标x2,y2
    img = results
    cls = int(x[-1])
    color = random.choice(colors)#随机选择一个颜色，用于后面画方框的颜色
    label = "{0}".format(classes[cls])#label为这个框所含目标类别名字的字符串
    cv2.rectangle(img, c1, c2,color, 1)# 在图片上画出(x1,y1,x2,y2)矩形，即我们检测到的目标方框
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]# 得到一个包含目标名字字符的方框的宽高
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4# 得到包含目标名字的方框右下角坐标c2，这里在x,y方向上分别加了3、4个像素
    cv2.rectangle(img, c1, c2,color, -1)# 在图片上画一个实心方框，我们将在方框内放置目标类别名字
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);# 在图片上写文字，(c1[0], c1[1] + t_size[1] + 4)为字符串的左下角坐标
    return img


#Detection phase

videofile = args.videofile #or path to the video file. 

cap = cv2.VideoCapture(videofile) #用 OpenCV 打开视频

#cap = cv2.VideoCapture(0)  #for webcam(相机)

# 当没有打开视频时抛出错误
assert cap.isOpened(), 'Cannot capture source'
# frames用于统计图片的帧数
frames = 0  
start = time.time()

fourcc = cv2.VideoWriter_fourcc('M','J','P','G') 
fps = 24 
savedPath = './det/savevideo.avi' # 保存的地址和视频名
ret, frame = cap.read() 
videoWriter = cv2.VideoWriter(savedPath, fourcc, fps,(frame.shape[1], frame.shape[0])) # 最后为视频图片的形状

while cap.isOpened():# ret指示是否读入了一张图片，为true时读入了一帧图片
    ret, frame = cap.read()
    
    if ret:
        # 将图片按照比例缩放缩放，将空白部分用(128,128,128)填充，得到为416x416的图片。并且将HxWxC转换为CxHxW   
        img = prep_image(frame, inp_dim)
        #cv2.imshow("a", frame)
        # 得到图片的W,H,是一个二元素tuple.因为我们不必再处理 batch，而是一次只处理一张图像，所以很多地方的代码都进行了简化。
        #因为一次只处理一帧，故使用一个元组im_dim替代 im_dim_list 的张量。
        im_dim = frame.shape[1], frame.shape[0]
#先将im_dim变成长度为2的一维行tensor，再在1维度(列这个维度)上复制一次，变成1x4的二维行tensor[W,H,W,H]，展开成1x4主要是在后面计算x1,y1,x2,y2各自对应的缩放系数时好对应上。  
        im_dim = torch.FloatTensor(im_dim).repeat(1,2)#repeat()可能会改变tensor的维度。它对tensor中对应repeat参数对应的维度上进行重复给定的次数，如果tensor的维度小于repeat()参数给定的维度，tensor的维度将变成和repeat()一致。这里repeat(1,2)，表示在第一维度上重复一次，第二维上重复两次，repeat(1,2)有2个元素，表示它给定的维度有2个,所以将长度为2的一维行tensor变成了维度为1x4的二维tensor   
                
        if CUDA:
            im_dim = im_dim.cuda()
            img = img.cuda()
        # 只进行前向计算，不计算梯度
        with torch.no_grad():
#得到每个预测方框在输入网络图片(416x416)坐标系中的坐标和宽高以及目标得分以及各个类别得分(x,y,w,h,s,s_cls1,s_cls2...)
#并且将tensor的维度转换成(batch_size, grid_size*grid_size*num_anchors, 5+类别数量)
            output = model(Variable(img, volatile = True), CUDA)
        #将方框属性转换成(ind,x1,y1,x2,y2,s,s_cls,index_cls)，去掉低分，NMS等操作，得到在输入网络坐标系中的最终预测结果
        output = write_results(output, confidence, num_classes, nms_conf = nms_thesh)

        # output的正常输出类型为float32,如果没有检测到目标时output元素为0，此时为int型，将会用continue进行下一次检测
        if type(output) == int:
        #每次迭代，我们都会跟踪名为frames的变量中帧的数量。然后我们用这个数字除以自第一帧以来过去的时间，得到视频的帧率。
            frames += 1
            print("FPS of the video is {:5.4f}".format( frames / (time.time() - start)))
        #我们不再使用cv2.imwrite将检测结果图像写入磁盘，而是使用cv2.imshow展示画有边界框的帧。
            cv2.imshow("frame", frame)
            key = cv2.waitKey(1)
        #如果用户按Q按钮，就会让代码中断循环，并且视频终止。
            if key & 0xFF == ord('q'):
                break
            continue
        
#im_dim一行对应一个方框所在图片尺寸。在detect.py中一次测试多张图片，所以对应的im_dim_list是找到每个方框对应的图片的尺寸。
# 而这里每次只有一张图片，每个方框所在图片的尺寸一样，只需将图片的尺寸的行数重复方框的数量次数即可                
        im_dim = im_dim.repeat(output.size(0), 1)
        # 得到每个方框所在图片缩放系数
        #scaling_factor = torch.min(416/im_dim,1)[0].view(-1,1)#这是源代码，下面是我修改的代码
        scaling_factor = torch.min(int(args.reso)/im_dim,1)[0].view(-1,1)
        # 将方框的坐标(x1,y1,x2,y2)转换为相对于填充后的图片中包含原始图片区域（如416*312区域）的计算方式。
        output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim[:,0].view(-1,1))/2
        output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim[:,1].view(-1,1))/2
        # 将坐标映射回原始图片
        output[:,1:5] /= scaling_factor
        #将超过了原始图片范围的方框坐标限定在图片范围之内
        for i in range(output.shape[0]):
            output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim[i,0])
            output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim[i,1])
    
        
        #coco.names文件中保存的是所有类别的名字，load_classes()返回一个列表classes，每个元素是一个类别的名字
        classes = load_classes('data/coco.names')
        #读入包含100个颜色的文件pallete，里面是100个三元组序列
        colors = pkl.load(open("pallete", "rb"))
        #将每个方框的属性写在图片上
        list(map(lambda x: write(x, frame), output))
        
        cv2.imshow("frame", frame)
        
        videoWriter.write(frame)           # 每次循环，写入该帧
        key = cv2.waitKey(1)
        # 如果有按键输入则返回按键值编码，输入q返回113
        if key & 0xFF == ord('q'):
            break
        #统计已经处理过的帧数
        frames += 1
        print(time.time() - start)
        print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
    else:
        videoWriter.release()              # 结束循环的时候释放
        break     
```

