# R-CenterNet(中文)<English readme is below>
基于CenterNet的旋转目标检测

### 前言
本工作初衷是提供一个**极其精简**的CenterNet代码，并对旋转目标进行检测，1.0为：
 ~~~
  ${R-CenterNet_ROOT}
  |-- backbone
  `-- |-- dlanet.py
      |-- dlanet_dcn.py
  |-- Loss.py
  |-- dataset.py
  |-- train.py
  |-- predict.py
 ~~~
应读者需求，随后更新了2.0
 ~~~
  ${R-CenterNet_ROOT}
  |-- labelGenerator
  `-- |-- Annotations
      |-- voc2coco.py
  |-- evaluation.py
 ~~~
 2.0以及data/airplane、imgs、ret文件夹都不是必须的，如果您只是想快速上手，1.0足够了。
 
#### demo
* R-DLADCN(推荐)(DCN编译与原版[CenterNet](https://github.com/xingyizhou/centernet)保持一致)
    * ![image](ret/R-DLADCN.jpg)
* R-ResDCN(主干网用的ResNet而不是DLA)
    * ![image](ret/R-ResDCN.jpg)
* R-DLANet(如果你不会编译DCN，就使用这个没有编译DCN的主干网)
    * ![image](ret/R-DLANet.jpg)
* DLADCN.jpg
    * ![image](ret/DLADCN.jpg)

#### 常见问题
 * 我对CenterNet[原版代码](https://github.com/xingyizhou/centernet) 进行了重构，使代码看起来更加简洁。
 * 如何编译DCN以及环境需求, 与[CenterNet](https://github.com/xingyizhou/centernet) 原版保持一致。
 * 关于数据处理与更多细节, 可以参考 [here](https://zhuanlan.zhihu.com/p/163696749)
 * torch版本1.2，如果你用的0.4会发生报错。
#### 训练自己的多分类网络
 * 打标签用labelGenerator文件夹里面的代码。
 * 修改代码中所有num_classes为你的类别数目，并且修改back_bone中hm的数目为你的类别数，如：
   def DlaNet(num_layers=34, heads = {'hm': your classes num, 'wh': 2, 'ang':1, 'reg': 2}, head_conv=256, plot=False):

#### Related projects
* [CenterNet](https://github.com/xingyizhou/centernet)



# R-CenterNet(English)
detector for rotated-objections based on CenterNet

### preface
The original intention of this work is to provide a **extremely compact** code of CenterNet and detect rotating targets: 1.0
 ~~~
  ${R-CenterNet_ROOT}
  |-- backbone
  `-- |-- dlanet.py
      |-- dlanet_dcn.py
  |-- Loss.py
  |-- dataset.py
  |-- train.py
  |-- predict.py
 ~~~
At the request of readers, 2.0 was subsequently updated：2.0
 ~~~
  ${R-CenterNet_ROOT}
  |-- labelGenerator
  `-- |-- Annotations
      |-- voc2coco.py
  |-- evaluation.py
 ~~~
 2.0 and the data/airplane, imgs, ret folders are not required. If you just want to get started quickly, 1.0 is enough。

#### demo
* R-DLADCN(this code)(How to complie dcn refer to the original code of [CenterNet](https://github.com/xingyizhou/centernet))
    * ![image](ret/R-DLADCN.jpg)
* R-ResDCN(just replace cnn in resnet with dcn)
    * ![image](ret/R-ResDCN.jpg)
* R-DLANet(not use dcn if you don't know how to complie dcn)
    * ![image](ret/R-DLANet.jpg)
* DLADCN.jpg
    * ![image](ret/DLADCN.jpg)

#### notes
 * I refactored the original [code](https://github.com/xingyizhou/centernet) to make codes more concise.
 * How to complie dcn and configure the environment, refer to the original code of [CenterNet](https://github.com/xingyizhou/centernet).
 * For data processing and more details, refer to [here](https://zhuanlan.zhihu.com/p/163696749)
 * torch version==1.2，don't use version==0.4!
#### train your data
 * label your data use labelGenerator;
 * modify all num_classes to your classes num, and modify the num of hm in your back_bone, such as:
   def DlaNet(num_layers=34, heads = {'hm': your classes num, 'wh': 2, 'ang':1, 'reg': 2}, head_conv=256, plot=False):

#### Related projects
* [CenterNet](https://github.com/xingyizhou/centernet)
