import torch
import utils
import numpy as np
# from apex import amp
from tqdm import tqdm
import torch.nn as nn
from block import Choice_Block, Choice_Block_x


channel = [16,
           64, 64, 64, 64,
           160, 160, 160, 160,
           320, 320, 320, 320, 320, 320, 320, 320,
           640, 640, 640, 640]
last_channel = 1024


class SinglePath_OneShot(nn.Module):
    def __init__(self, dataset, resize, classes, layers):
        super(SinglePath_OneShot, self).__init__()
        if dataset == 'cifar10' and not resize:
            first_stride = 1
            self.downsample_layers = [4, 8]
        elif dataset == 'imagenet' or resize:
            first_stride = 2
            self.downsample_layers = [0, 4, 8, 16]
        self.classes = classes
        self.layers = layers
        self.kernel_list = [3, 5, 7, 'x']

        # stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, channel[0], kernel_size=3, stride=first_stride, padding=1, bias=False),
            nn.BatchNorm2d(channel[0], affine=False),
            nn.ReLU6(inplace=True)
        )
        # choice_block
        self.choice_block = nn.ModuleList([])
        for i in range(layers):
            if i in self.downsample_layers:
                stride = 2
                inp, oup = channel[i], channel[i + 1]
            else:
                stride = 1
                inp, oup = channel[i] // 2, channel[i + 1]
            layer_cb = nn.ModuleList([])
            for j in self.kernel_list:
                if j == 'x':
                    layer_cb.append(Choice_Block_x(inp, oup, stride=stride))
                else:
                    layer_cb.append(Choice_Block(inp, oup, kernel=j, stride=stride))
            self.choice_block.append(layer_cb)
        # last_conv
        self.last_conv = nn.Sequential(
            nn.Conv2d(channel[-1], last_channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(last_channel, affine=False),
            nn.ReLU6(inplace=True)
        )
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        # self.global_pooling = nn.AvgPool2d(7)
        self.classifier = nn.Linear(last_channel, self.classes, bias=False)
        self._initialize_weights()

    def forward(self, x, choice=np.random.randint(4, size=20)):
        x = self.stem(x)
        # repeat
        for i, j in enumerate(choice):
            x = self.choice_block[i][j](x)
        x = self.last_conv(x)
        x = self.global_pooling(x)
        x = x.view(-1, last_channel)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class SinglePath_Network(nn.Module):
    def __init__(self, dataset, resize, classes, layers, choice):
        super(SinglePath_Network, self).__init__()
        if dataset == 'cifar10' and not resize:
            first_stride = 1
            self.downsample_layers = [4, 8]
        elif dataset == 'imagenet' or resize:
            first_stride = 2
            self.downsample_layers = [0, 4, 8, 16]
        self.classes = classes
        self.layers = layers
        self.kernel_list = [3, 5, 7, 'x']

        # stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, channel[0], kernel_size=3, stride=first_stride, padding=1, bias=False),
            nn.BatchNorm2d(channel[0]),
            nn.ReLU6(inplace=True)
        )
        # choice_block
        self.choice_block = nn.ModuleList([])
        for i in range(layers):
            if i in self.downsample_layers:
                stride = 2
                inp, oup = channel[i], channel[i + 1]
            else:
                stride = 1
                inp, oup = channel[i] // 2, channel[i + 1]
            if choice[i] == 3:
                self.choice_block.append(Choice_Block_x(inp, oup, stride=stride, supernet=False))
            else:
                self.choice_block.append(Choice_Block(inp, oup, kernel=self.kernel_list[choice[i]], stride=stride, supernet=False))
        # last_conv
        self.last_conv = nn.Sequential(
            nn.Conv2d(channel[-1], last_channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(last_channel),
            nn.ReLU6(inplace=True)
        )
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        # self.global_pooling = nn.AvgPool2d(7)
        self.classifier = nn.Linear(last_channel, self.classes, bias=False)
        self._initialize_weights()

    def forward(self, x):
        x = self.stem(x)
        # repeat
        for i in range(self.layers):
            x = self.choice_block[i](x)
        x = self.last_conv(x)
        x = self.global_pooling(x)
        x = x.view(-1, last_channel)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


def train(args, epoch, train_data, device, model, criterion, optimizer, scheduler, supernet):
    model.train()
    train_loss = 0.0
    top1 = utils.AvgrageMeter()
    train_data = tqdm(train_data)
    train_data.set_description('[%s%04d/%04d %s%f]' % ('Epoch:', epoch + 1, args.epochs, 'lr:', scheduler.get_lr()[0]))
    for step, (inputs, targets) in enumerate(train_data):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        if supernet:
            # layers=20 num_choices=4 
            choice = utils.random_choice(args.num_choices, args.layers)
            outputs = model(inputs, choice)
        else:
            outputs = model(inputs)
        loss = criterion(outputs, targets)
        # if args.dataset == 'cifar10':
        loss.backward()
        # elif args.dataset == 'imagenet':
        #     with amp.scale_loss(loss, optimizer) as scaled_loss:
        #         scaled_loss.backward()
        optimizer.step()
        prec1, prec5 = utils.accuracy(outputs, targets, topk=(1, 5))
        n = inputs.size(0)
        top1.update(prec1.item(), n)
        train_loss += loss.item()
        postfix = {'train_loss': '%.6f' % (train_loss / (step + 1)), 'train_acc': '%.6f' % top1.avg}
        train_data.set_postfix(log=postfix)


def validate(args, epoch, val_data, device, model, criterion, supernet, choice=None):
    model.eval()
    val_loss = 0.0
    val_top1 = utils.AvgrageMeter()
    with torch.no_grad():
        for step, (inputs, targets) in enumerate(val_data):
            inputs, targets = inputs.to(device), targets.to(device)
            if supernet:
                if choice == None:
                    choice = utils.random_choice(args.num_choices, args.layers)
                outputs = model(inputs, choice)
            else:
                outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            prec1, prec5 = utils.accuracy(outputs, targets, topk=(1, 5))
            n = inputs.size(0)
            val_top1.update(prec1.item(), n)
        print('[Val_Accuracy epoch:%d] val_loss:%f, val_acc:%f'
              % (epoch + 1, val_loss / (step + 1), val_top1.avg))
        return val_top1.avg
