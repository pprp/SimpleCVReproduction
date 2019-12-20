import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import init
from torchvision import models

# class ClassBlock(nn.Module):
#     def __init__(self, input_dim, class_num, dropout=False, relu=False):
#         super(ClassBlock, self).__init__()
#         # base block
#         blocks = []
#         blocks.append(nn.BatchNorm1d(input_dim))
#         if relu:
#             blocks.append(nn.ReLU())
#         if dropout:
#             blocks.append(nn.Dropout(0.5))
#         self.base = nn.Sequential(*blocks)

#         # classifier
#         self.classifier = nn.Sequential(nn.Linear(input_dim, class_num))

#     def forward(self, x):
#         x = self.base(x)
#         # apply L2 norm
#         x_norm = x.norm(p=2, dim=1, keepdim=True) + 1e-8
#         f = x.div(x_norm)
#         x = self.classifier(f)
#         return x, f


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_out')
        init.constant(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal(m.weight.data, std=0.001)
        init.constant(m.bias.data, 0.0)


class ClassBlock(nn.Module):
    def __init__(self,
                 input_dim,
                 class_num,
                 dropout=False,
                 relu=False,
                 num_bottleneck=512):
        super(ClassBlock, self).__init__()
        add_block = []
        #add_block += [nn.Linear(input_dim, num_bottleneck)]
        num_bottleneck = input_dim
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if dropout:
            add_block += [nn.Dropout(p=0.5)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        f = self.add_block(x)
        f_norm = f.norm(p=2, dim=1, keepdim=True) + 1e-8
        f = f.div(f_norm)
        x = self.classifier(f)
        return x, f


class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18, self).__init__()
        self.model = models.resnet50(pretrained=False)
        self.model.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = ClassBlock(2048,
                                     num_classes,
                                     dropout=False,
                                     relu=False)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = torch.squeeze(x)
        x, f = self.classifier(x)
        return x, f

    def save(self, file_path):
        if file_path is not None:
            torch.save(self.state_dict(), file_path)

    def load(self, weight_path):
        if weight_path is not None:
            self.load_state_dict(torch.load(weight_path))
            print("Loading %s sucessfully!" % weight_path)
        else:
            print("Occur errors in loading %s" % weight_path)
