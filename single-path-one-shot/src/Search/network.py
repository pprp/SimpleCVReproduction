import torch
import torch.nn as nn
from blocks import Shufflenet, Shuffle_Xception


class ShuffleNetV2_OneShot(nn.Module):

    def __init__(self, input_size=224, n_class=1000):
        super(ShuffleNetV2_OneShot, self).__init__()

        assert input_size % 32 == 0

        self.stage_repeats = [4, 4, 8, 4]
        self.stage_out_channels = [-1, 16, 64, 160, 320, 640, 1024]

        # building first layer
        input_channel = self.stage_out_channels[1]
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel, affine=False),
            nn.ReLU(inplace=True),
        )

        self.features = torch.nn.ModuleList()
        archIndex = 0
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage + 2]

            for i in range(numrepeat):
                if i == 0:
                    inp, outp, stride = input_channel, output_channel, 2
                else:
                    inp, outp, stride = input_channel // 2, output_channel, 1

                base_mid_channels = outp // 2
                mid_channels = int(base_mid_channels)
                archIndex += 1
                self.features.append(torch.nn.ModuleList())
                for blockIndex in range(4):
                    if blockIndex == 0:
                        print('Shuffle3x3')
                        self.features[-1].append(
                            Shufflenet(inp, outp, mid_channels=mid_channels, ksize=3, stride=stride))
                    elif blockIndex == 1:
                        print('Shuffle5x5')
                        self.features[-1].append(
                            Shufflenet(inp, outp, mid_channels=mid_channels, ksize=5, stride=stride))
                    elif blockIndex == 2:
                        print('Shuffle7x7')
                        self.features[-1].append(
                            Shufflenet(inp, outp, mid_channels=mid_channels, ksize=7, stride=stride))
                    elif blockIndex == 3:
                        print('Xception')
                        self.features[-1].append(
                            Shuffle_Xception(inp, outp, mid_channels=mid_channels, stride=stride))
                    else:
                        raise NotImplementedError
                input_channel = output_channel

        self.archLen = archIndex
        # self.features = nn.Sequential(*self.features)

        self.conv_last = nn.Sequential(
            nn.Conv2d(
                input_channel, self.stage_out_channels[
                    -1], 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.stage_out_channels[-1], affine=False),
            nn.ReLU(inplace=True),
        )
        self.globalpool = nn.AvgPool2d(7)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Sequential(
            nn.Linear(self.stage_out_channels[-1], n_class, bias=False))
        self._initialize_weights()

    def forward(self, x, architecture):
        assert self.archLen == len(architecture)

        x = self.first_conv(x)

        for archs, arch_id in zip(self.features, architecture):
            x = archs[arch_id](x)

        x = self.conv_last(x)

        x = self.globalpool(x)

        x = self.dropout(x)
        x = x.contiguous().view(-1, self.stage_out_channels[-1])
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

if __name__ == "__main__":
    # architecture = [0, 0, 3, 1, 1, 1, 0, 0, 2, 0, 2, 1, 1, 0, 2, 0, 2, 1, 3, 2]
    # scale_list = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]
    # scale_ids = [6, 5, 3, 5, 2, 6, 3, 4, 2, 5, 7, 5, 4, 6, 7, 4, 4, 5, 4, 3]
    channels_scales = []
    for i in range(len(scale_ids)):
        channels_scales.append(scale_list[scale_ids[i]])
    model = ShuffleNetV2_OneShot()
    # print(model)

    test_data = torch.rand(5, 3, 224, 224)
    test_outputs = model(test_data)
    print(test_outputs.size())
