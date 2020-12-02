# 标签说明
1.先用这个[工具](https://link.zhihu.com/?target=https%3A//github.com/chinakook/labelImg2)对你自己的数据打标签对每个图片输出一个.xml文件
	

    <object>
    	<robndbox>
    		<cx>602.1491</cx>
    		<cy>523.8509</cy>
    		<w>93.0</w>
    		<h>105.0</h>
    		<angle>0.54</angle>
    	</robndbox>
    </object>
	
其中角度是以12点钟为0°，顺时针方向，最大值为179.99999°（旋转180°，相当于没转），这里是用π来表示的。

2.用PascalVOC2coco将你打完标签的所有.xml文件合并成一个json文件，也就是R-CenterNet需要的训练数据
由于loss计算时候，角度损失是用180°角度体系计算的，所以会将上面.xml文件中π角度转化成180°
![image](https://pic2.zhimg.com/80/v2-b34dac0e5256cd81d6f0a008cc77308d_720w.jpg)


# instructions
1.[工具](https://link.zhihu.com/?target=https%3A//github.com/chinakook/labelImg2)label your images, one image will output one .xml file


    <object>
    	<robndbox>
    		<cx>602.1491</cx>
    		<cy>523.8509</cy>
    		<w>93.0</w>
    		<h>105.0</h>
    		<angle>0.54</angle>
    	</robndbox>
    </object>
	
Where the Angle is 0° at 12 o 'clock, clockwise, with a maximum of 179.99999° (rotated 180°, equivalent to no rotation), this is represented in terms of π.

2.use this function PascalVOC2coco to convert all "*.xml" file label produced by 1. to a .josn file, also is train data of R-CenterNet.
π is going to be converted into 180, and compute angle loss.
![image](https://pic2.zhimg.com/80/v2-b34dac0e5256cd81d6f0a008cc77308d_720w.jpg)
