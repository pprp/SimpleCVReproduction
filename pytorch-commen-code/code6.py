# 计算模型整体参数量

num_parameters = sum(torch.numel(parameter) for parameter in model.parameters())

# 查看网络中的参数
# 可以通过model.state_dict()或者model.named_parameters()函数查看现在的全部可训练参数（包括通过继承得到的父类中的参数）

params = list(model.named_parameters())
(name, param) = params[28]
print(name)
print(param.grad)
print('-------------------------------------------------')
(name2, param2) = params[29]
print(name2)
print(param2.grad)
print('----------------------------------------------------')
(name1, param1) = params[30]
print(name1)
print(param1.grad)

# 模型可视化（使用pytorchviz）
# 类似 Keras 的 model.summary() 输出模型信息（使用pytorch-summary ）

# 模型权重初始化

# 注意 model.modules() 和 model.children() 的区别：
# model.modules() 会迭代地遍历模型的所有子层，
# 而 model.children() 只会遍历模型下的一层。

# Common practise for initialization.
for layer in model.modules():
    if isinstance(layer, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(layer.weight, mode='fan_out',
                                      nonlinearity='relu')
        if layer.bias is not None:
            torch.nn.init.constant_(layer.bias, val=0.0)
    elif isinstance(layer, torch.nn.BatchNorm2d):
        torch.nn.init.constant_(layer.weight, val=1.0)
        torch.nn.init.constant_(layer.bias, val=0.0)
    elif isinstance(layer, torch.nn.Linear):
        torch.nn.init.xavier_normal_(layer.weight)
        if layer.bias is not None:
            torch.nn.init.constant_(layer.bias, val=0.0)

# Initialization with given tensor.
layer.weight = torch.nn.Parameter(tensor)

# 提取模型中的某一层
# modules()会返回模型中所有模块的迭代器，它能够访问到最内层，
# 比如self.layer1.conv1这个模块，还有一个与它们相对应的是name_children()属性
# 以及named_modules(),这两个不仅会返回模块的迭代器，还会返回网络层的名字。


# 取模型中的前两层
new_model = nn.Sequential(*list(model.children())[:2] 
# 如果希望提取出模型中的所有卷积层，可以像下面这样操作：
for layer in model.named_modules():
    if isinstance(layer[1],nn.Conv2d):
         conv_model.add_module(layer[0],layer[1])

