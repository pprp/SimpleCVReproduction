# 关键点归回模型

使用的标注工具是： https://github.com/pprp/landmark_annotation 

![](https://github.com/pprp/landmark_annotation/raw/master/README.assets/1586482369201.png)

数据：每张图一个关键点，100张小数据集

思路：图片经过卷积神经网络得到一个点的坐标

训练过程:

使用SmoothL1Loss训练：

![](readme.assets/1586505357127.png)

使用MSELoss训练：

![1586512915230](readme.assets/1586512915230.png)

效果：由于数据集比较小，效果还可以，不过对于目标过小的情况下，效果不好。由于数据集是非公开的，所以就不展示了。