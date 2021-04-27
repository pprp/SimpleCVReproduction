import math
import paddle
import paddle.nn as nn
import numpy as np
import random 
from paddle import callbacks
from paddle.nn import CrossEntropyLoss

__all__ = ['ResNet20', 'ResNet56', 'ResNet110']

##########   Original_Module   ##########
class Block1(nn.Layer):
    def __init__(self, in_planes, places, stride=1):
        super().__init__()
        self.conv1_input_channel = in_planes
        self.output_channel = places

        #defining conv1
        self.conv1 = self.Conv(self.conv1_input_channel, self.output_channel, kernel_size=3, stride=stride, padding=1)

    def Conv(self, in_places, places, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2D(in_channels=in_places, out_channels=places,
                      kernel_size=kernel_size, stride=stride, padding=padding, bias_attr=False),
            nn.BatchNorm2D(places),
            nn.ReLU())

    def forward(self, x):
        out = self.conv1(x)
        return out


class BasicBolock(nn.Layer):
    def __init__(self, len_list, stride=1, group=1, downsampling=False):
        super(BasicBolock,self).__init__()
        global IND

        self.downsampling = downsampling
        self.adaptive_pooling = False
        self.len_list = len_list

        self.conv1 = self.Conv(self.len_list[IND-1], self.len_list[IND], kernel_size=3, stride=stride, padding=1)
        self.conv2 = self.Conv(self.len_list[IND], self.len_list[IND+1], kernel_size=3, stride=1, padding=1)

        if self.downsampling :
            self.downsample = nn.Sequential(
            nn.Conv2D(in_channels=self.len_list[IND-1], out_channels=self.len_list[IND+1], kernel_size=1, stride=stride, padding=0, bias_attr=False),
            nn.BatchNorm2D(self.len_list[IND+1]))
        elif not self.downsampling and (self.len_list[IND-1] != self.len_list[IND+1]):
            self.downsample = nn.Sequential(
            nn.Conv2D(in_channels=self.len_list[IND-1], out_channels=self.len_list[IND+1], kernel_size=1, stride=stride, padding=0, bias_attr=False),
            nn.BatchNorm2D(self.len_list[IND+1]))
            self.downsampling = True
        self.relu = nn.ReLU()
        IND += 2

    def Conv(self, in_places, places, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2D(in_channels=in_places, out_channels=places, kernel_size=kernel_size, stride=stride, padding=padding, bias_attr=False),
            nn.BatchNorm2D(places))

    def forward(self, x):
        residual = x
        x = self.relu(self.conv1(x))
        out = self.conv2(x)
        if self.downsampling:
            residual = self.downsample(residual)
        out += residual
        out = self.relu(out)
        return out


def _calculate_fan_in_and_fan_out(tensor, op):
    op = op.lower()
    valid_modes = ['linear', 'conv']
    if op not in valid_modes:
        raise ValueError("op {} not supported, please use one of {}".format(op, valid_modes))
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
    if op == 'linear':
        num_input_fmaps = tensor.shape[0]
        num_output_fmaps = tensor.shape[1]
    else:
        num_input_fmaps = tensor.shape[1]
        num_output_fmaps = tensor.shape[0]
    receptive_field_size = 1
    if tensor.dim() > 2:
        receptive_field_size = tensor[0][0].size
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


def _calculate_correct_fan(tensor, op, mode):
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:
        raise ValueError("Mode {} not supported, please use one of {}".format(mode, valid_modes))

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor, op)
    return fan_in if mode == 'fan_in' else fan_out


def kaiming_normal_(tensor, op='linear', a=0, mode='fan_in', nonlinearity='leaky_relu'):
    fan = _calculate_correct_fan(tensor, op, mode)
    gain = math.sqrt(2.0)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    with paddle.no_grad():
        return paddle.assign(paddle.uniform(tensor.shape, min=-bound, max=bound), tensor)


class ResNet(nn.Layer):
    def __init__(self, blocks, len_list, module_type=BasicBolock, num_classes=10, expansion=1):
        super(ResNet,self).__init__()
        self.block = module_type
        self.len_list = len_list
        self.expansion = expansion

        global IND
        IND = 0

        self.conv1 = Block1(in_planes=3, places=self.len_list[IND])  # 1
        IND += 1
        self.layer1 = self.make_layer(self.len_list, block=blocks[0], block_type=self.block, stride=1)
        self.layer2 = self.make_layer(self.len_list, block=blocks[1], block_type=self.block, stride=2)
        self.layer3 = self.make_layer(self.len_list, block=blocks[2], block_type=self.block, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2D((1, 1))
        self.fc = nn.Linear(self.len_list[-2], num_classes)

        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                kaiming_normal_(m.weight, op='conv', mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2D):
                paddle.assign(paddle.ones(m.weight.shape), m.weight)
                paddle.assign(paddle.zeros(m.bias.shape), m.bias)
            elif isinstance(m, nn.Linear):
                kaiming_normal_(m.weight, op='linear', mode='fan_out', nonlinearity='relu')
                paddle.assign(paddle.zeros(m.bias.shape), m.bias)

    def make_layer(self, len_list, block, block_type, stride):
        layers = []
        layers.append(block_type(len_list, stride, downsampling =True))
        for i in range(1, block):
            layers.append(block_type(len_list))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        x = self.avgpool(out3).flatten(1)
        x = self.fc(x)
        return x


##########   Different ResNet Model   ##########
#default block type --- BasicBolock for ResNet20/56/110;
def ResNet20(CLASS, len_list=None):
    return ResNet([3, 3, 3], len_list=len_list, num_classes=CLASS, module_type=BasicBolock)

def ResNet56(CLASS, len_list=None):
    return ResNet([9, 9, 9], len_list=len_list, num_classes=CLASS, module_type=BasicBolock)

def ResNet110(CLASS, len_list=None):
    return ResNet([18, 18, 18], len_list=len_list, num_classes=CLASS, module_type=BasicBolock)


class ToArray(object):
    def __call__(self, img):
        img = np.array(img)
        img = np.transpose(img, [2, 0, 1])
        img = img / 255.
        return img.astype('float32')


class RandomApply(object):
    def __init__(self, transform, p=0.5):
        super().__init__()
        self.p = p
        self.transform = transform

    def __call__(self, img):
        if self.p < random.random():
            return img
        img = self.transform(img)
        return img

class LRSchedulerM(callbacks.LRScheduler):
    def __init__(self, by_step=False, by_epoch=True, warm_up=True):
        super().__init__(by_step, by_epoch)
        assert by_step ^ warm_up
        self.warm_up = warm_up

    def on_epoch_end(self, epoch, logs=None):
        if self.by_epoch and not self.warm_up:
            if self.model._optimizer and hasattr(
                self.model._optimizer, '_learning_rate') and isinstance(
                    self.model._optimizer._learning_rate, paddle.optimizer.lr.LRScheduler):
                self.model._optimizer._learning_rate.step()

    def on_train_batch_end(self, step, logs=None):
        if self.by_step or self.warm_up:
            if self.model._optimizer and hasattr(
                self.model._optimizer, '_learning_rate') and isinstance(
                    self.model._optimizer._learning_rate, paddle.optimizer.lr.LRScheduler):
                self.model._optimizer._learning_rate.step()
            if self.model._optimizer._learning_rate.last_epoch >= self.model._optimizer._learning_rate.warmup_steps:
                self.warm_up = False


def _on_train_batch_end(self, step, logs=None):
    logs = logs or {}
    logs['lr'] = self.model._optimizer.get_lr()
    self.train_step += 1
    if self._is_write():
        self._updates(logs, 'train')

def _on_train_begin(self, logs=None):
    self.epochs = self.params['epochs']
    assert self.epochs
    self.train_metrics = self.params['metrics'] + ['lr']
    assert self.train_metrics
    self._is_fit = True
    self.train_step = 0

callbacks.VisualDL.on_train_batch_end = _on_train_batch_end
callbacks.VisualDL.on_train_begin = _on_train_begin


def export_static_model(model, model_path, input_dtype='float32'):
    input_shape = [
        paddle.static.InputSpec(
            shape=[2, 3, 32, 32], dtype=input_dtype),
    ]
    net = paddle.jit.to_static(model, input_spec=input_shape)
    paddle.jit.save(net, model_path)


def _loss_forward(self, input, tea_input, label=None):
    if label is not None:
        ret = paddle.nn.functional.cross_entropy(
            input,
            label,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            soft_label=self.soft_label,
            axis=self.axis,
            name=self.name)

        mse = paddle.nn.functional.cross_entropy(
            input,
            paddle.nn.functional.softmax(tea_input),
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            soft_label=True,
            axis=self.axis)
        # mse = paddle.nn.functional.mse_loss(input, tea_input)
        return ret, mse
    else:
        ret = paddle.nn.functional.cross_entropy(
            input,
            tea_input,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            soft_label=self.soft_label,
            axis=self.axis,
            name=self.name)
        return ret
CrossEntropyLoss.forward = _loss_forward