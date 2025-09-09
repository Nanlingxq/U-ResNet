# Here is the code :

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape: b, num_channels, h, w  -->  b, groups, channels_per_group, h, w
    x = x.view(batchsize, groups, channels_per_group, height, width)

    # channelshuffle
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class shuffleNet_unit(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride, groups):
        super(shuffleNet_unit, self).__init__()

        mid_channels = out_channels//4
        self.stride = stride
        if in_channels == 24:
            self.groups = 1
        else:
            self.groups = groups
        self.GConv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=1, groups=self.groups, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        self.DWConv = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=self.stride, padding=1, groups=self.groups, bias=False),
            nn.BatchNorm2d(mid_channels)
        )

        self.GConv2 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=1, groups=self.groups, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        if self.stride == 2:
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = self.GConv1(x)
        out = channel_shuffle(out, groups=self.groups)
        out = self.DWConv(out)
        out = self.GConv2(out)
        short = self.shortcut(x)
        if self.stride == 2:
            out = F.relu(torch.cat([out, short], dim=1))
        else:
            out = F.relu(out + short)
        return out


class ShuffleNet(nn.Module):
    def __init__(self, groups, num_layers, num_channels, num_classes=100):
        super(ShuffleNet, self).__init__()
        self.name = "ShuffleNet"
        self.groups = groups
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 24, 3, 2, 1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stage2 = self.make_layers(24, num_channels[0], num_layers[0], groups)
        self.stage3 = self.make_layers(num_channels[0], num_channels[1], num_layers[1], groups)
        self.stage4 = self.make_layers(num_channels[1], num_channels[2], num_layers[2], groups)

        self.globalpool = nn.AvgPool2d(kernel_size=3, stride=1)
        self.fc = nn.Linear(num_channels[2], num_classes)

    def make_layers(self, in_channels, out_channels, num_layers, groups):
        layers = []
        layers.append(shuffleNet_unit(in_channels, out_channels - in_channels, 2, groups))
        in_channels = out_channels
        for i in range(num_layers - 1):
            layers.append(shuffleNet_unit(in_channels, out_channels, 1, groups))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.globalpool(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out


def ShuffleNet_g1(**kwargs):
    num_layers = [4, 8, 4]
    num_channels = [144, 288, 576]
    model = ShuffleNet(1, num_layers, num_channels, **kwargs)
    return model


def ShuffleNet_g2(**kwargs):
    num_layers = [4, 8, 4]
    num_channels = [200, 400, 800]
    model = ShuffleNet(2, num_layers, num_channels, **kwargs)
    return model


def ShuffleNet_g3(**kwargs):
    num_layers = [4, 8, 4]
    num_channels = [240, 480, 960]
    model = ShuffleNet(3, num_layers, num_channels, **kwargs)
    return model


def ShuffleNet_g4(**kwargs):
    num_layers = [4, 8, 4]
    num_channels = [272, 544, 1088]
    model = ShuffleNet(4, num_layers, num_channels, **kwargs)
    return model


def ShuffleNet_g8(**kwargs):
    num_layers = [4, 8, 4]
    num_channels = [384, 768, 1536]
    model = ShuffleNet(8, num_layers, num_channels, **kwargs)
    return model


def test():
    net = ShuffleNet_g8()
    y = net(torch.randn(1, 3, 224, 224))
    print(y.size())
    summary(net, (1, 3, 224, 224), depth=5)
