import torch
import torch.nn as nn
import torch.nn.functional as F
class BasicBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(BasicBlock, self).__init__()
        self.name = "Basic-Block-ResNet"
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)

        return out

class Bottleneck(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(Bottleneck, self).__init__()
        self.name = 'Bottleneck'
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, int(outchannel / 4), kernel_size=1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(int(outchannel / 4)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(outchannel / 4), int(outchannel / 4), kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(int(outchannel / 4)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(outchannel / 4), outchannel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(outchannel),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        y = self.shortcut(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self, ResidulBlock, num_classes=10):
        super(ResNet18, self).__init__()
        self.name = "ResNet18"
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer1 = self.make_layer(ResidulBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResidulBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidulBlock, 256, 2, stride=1)
        self.layer4 = self.make_layer(ResidulBlock, 512, 2, stride=1)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self,x):
        out = self.conv1(x)
        # print(out.size())
        out = self.layer1(out)
        # print(out.size())
        out = self.layer2(out)
        # print(out.size())
        out = self.layer3(out)
        # print(out.size())
        out = self.layer4(out)
        # print(out.size())
        out = F.avg_pool2d(out, 7)
        # print(out.size())
        out = out.view(out.size(0), -1)
        # print(out.size())
        # print(out.size())
        out = self.fc(out)

        return out


class ResNet34(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10):
        super(ResNet34, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64, 3, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 4, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 6, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 3, stride=2)
        self.fc = nn.Linear(8192, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):  # 3*32*32
        out = self.conv1(x)  # 64*32*32
        out = self.layer1(out)  # 64*32*32
        out = self.layer2(out)  # 128*16*16
        out = self.layer3(out)  # 256*8*8
        out = self.layer4(out)  # 512*4*4
        out = F.avg_pool2d(out, 7)  # 512*1*1
        out = out.view(out.size(0), -1)  # 512
        out = self.fc(out)
        return out

class ResNet18UpSample(nn.Module):
    def __init__(self, ResidulBlock, num_classes=10, scale_factor=3):
        super(ResNet18UpSample, self).__init__()
        self.name = f"ResNet18UpSample-scale-{scale_factor}"
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode='bilinear'),
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer1 = self.make_layer(ResidulBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResidulBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidulBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidulBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512 * 4, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self,x):
        out = self.conv1(x)
        # print(out.size())
        out = self.layer1(out)
        # print(out.size())
        out = self.layer2(out)
        # print(out.size())
        out = self.layer3(out)
        # print(out.size())
        out = self.layer4(out)
        # print(out.size())
        out = F.avg_pool2d(out, 4)
        # print(out.size())
        out = out.view(out.size(0), -1)
        # print(out.size())
        # print(out.size())
        out = self.fc(out)

        return out

class ResNet18_EUCB(nn.Module):
    def __init__(self, ResidulBlock, num_classes=10, scale_factor=2):
        super(ResNet18_EUCB, self).__init__()
        self.name = f"ResNet18-EUCB-scale-{scale_factor}"
        self.scale_factor = scale_factor
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            EUCB(in_channels=3, out_channels=64, scale_factor=self.scale_factor),
            nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer1 = self.make_layer(ResidulBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResidulBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidulBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidulBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512 * 4, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        # layers.append(EUCB(in_channels=self.inchannel, out_channels=self.inchannel, scale_factor=self.scale_factor))
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self,x):
        out = self.conv1(x)
        # print(out.size())
        out = self.layer1(out)
        # print(out.size())
        out = self.layer2(out)
        # print(out.size())
        out = self.layer3(out)
        # print(out.size())
        out = self.layer4(out)
        # print(out.size())
        out = F.avg_pool2d(out, 7)
        # print(out.size())
        out = out.view(out.size(0), -1)
        # print(out.size())
        # print(out.size())
        out = self.fc(out)

        return out
