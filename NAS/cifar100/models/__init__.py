from __future__ import absolute_import
from models.xception import xception
from models.stochasticdepth import stochastic_depth_resnet101, stochastic_depth_resnet152, stochastic_depth_resnet18, stochastic_depth_resnet34, stochastic_depth_resnet50
from models.squeezenet import squeezenet

from .densenet import *
from .dla import *
from .dynamic_resnet20 import *
from .efficientnetb0 import *
from .googlenet import *
from .lenet import *
from .masked_resnet20 import *
from .mobilenet import *
from .mobilenetv2 import *
from .pnasnet import *
from .preact_resnet import *
from .regnet import *
from .resnet import *
from .resnet20 import *
from .resnext import *
from .sample_resnet20 import *
from .senet import *
from .shufflenet import *
from .shufflenetv2 import *
from .slimmable_resnet20 import *
from .supernet import *
from .vgg import *
from .attention import *
from .inceptionv3 import *
from .inceptionv4 import *
from .nasnet import *
from .rir import *
from .stochasticdepth import *
from .wideresidual import * 
from .xception import * 




__model_factory = {
    'dynamic': dynamic_resnet20,
    'masked': masked_resnet20,
    'resnet20': resnet20,
    'sample': sample_resnet20,
    'slimmable': slimmable_resnet20,
    'super': SuperNet,
    'densenet': densenet_cifar,
    'senet': senet18_cifar,
    'googlenet': GoogLeNet,
    'dla': DLA,
    'shufflenet': ShuffleNetG2,
    'shufflenetv2': ShuffleNetV2,
    'resnet18': ResNet18,
    'resnet34': ResNet34,
    'resnet50': ResNet50,
    'efficientnetb0': EfficientNetB0,
    'lenet': LeNet,
    'mobilenet': MobileNet,
    'mobilenetv2': MobileNetV2,
    'pnasnet': PNASNetB,
    'preact_resnet': PreActResNet18,
    'regnet': RegNetX_200MF,
    'resnext': ResNeXt29_2x64d,
    'vgg': vgg11,
    'attention56': attention56,
    'attention92': attention92,
    'inceptionv3': inceptionv3,
    'inceptionv4': inceptionv4,
    'inception_resnet_v2': inception_resnet_v2,
    'nasnet': nasnet,
    'rir': resnet_in_resnet,
    'squeezenet': squeezenet,
    'stochastic_depth_resnet18': stochastic_depth_resnet18,
    'stochastic_depth_resnet34': stochastic_depth_resnet34,
    'stochastic_depth_resnet50': stochastic_depth_resnet50,
    'stochastic_depth_resnet101': stochastic_depth_resnet101,
    'stochastic_depth_resnet152': stochastic_depth_resnet152,
    'wideresnet':wideresnet,
    'xception': xception
}


def show_available_models():
    """Displays available models

    """
    print(list(__model_factory.keys()))


def build_model(name, num_classes=100):
    avai_models = list(__model_factory.keys())
    if name not in avai_models:
        raise KeyError(
            'Unknown model: {}. Must be one of {}'.format(name, avai_models)
        )
    return __model_factory[name](num_classes=num_classes)
