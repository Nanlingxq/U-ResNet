import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1, padding=0):
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
    def __init__(self, inchannel, outchannel, stride=1, padding = 0):
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
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer1 = self.make_layer(ResidulBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResidulBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidulBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidulBlock, 512, 2, stride=2)
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
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        # print(out.size())
        out = self.fc(out)

        return out

class STNBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(STNBlock, self).__init__()
        self.name = 'Spatial Transformer Networks'
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(64, out_channels, kernel_size=5, padding=1)
        self.conv2_drop = nn.Dropout2d()


        self.localization = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, padding=2),
            nn.MaxPool2d(3, stride=2),
            nn.ReLU(True),
            nn.Conv2d(64, 16, kernel_size=5),
            nn.MaxPool2d(3, stride=2),
            nn.ReLU(True)
        )

        self.fc_loc = nn.Sequential(
            nn.Linear(64 * 4, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2),
        )

        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def stn(self, x):
        xs = self.localization(x)
        # print(f"xs_size1 =  {xs.size()}")
        xs = xs.view(-1,  64 * 4)
        # print(f"xs_size2 =  {xs.size()}")
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)


        # print(f"theta_size =  {theta.size()}")
        # print(f"x_size =  {x.size()}")

        grid = F.affine_grid(theta=theta, size=x.size(), align_corners=True)
        x = F.grid_sample(x, grid, align_corners=True)

        # print(f"STNBlock-Output = {x.size()}")
        return x

    def forward(self, x):
        x = self.stn(x)
        # print(f"x.size = {x.size()}")
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(self.conv2_drop(self.conv2(x)))
        # print(f"x1.size() = {x.size()}")
        x = F.dropout(x, training=self.training)
        x = self.conv3(x)
        nn.BatchNorm2d(64),
        nn.ReLU()
        # print(f"x1.size() = {x.size()}")
        return x

class STN_ResNet(nn.Module):
    def __init__(self, ResidulBlock, in_channels, num_classes):
        super(STN_ResNet, self).__init__()
        self.name = "STN_ResNet"
        self.inchannel = 64
        self.STN_Block = STNBlock(in_channels=in_channels, out_channels=64)
        self.layer1 = self.make_layer(ResidulBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResidulBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidulBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidulBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride, padding):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.STN_Block(x)
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
        out = self.fc(out)

        return out

if __name__ == '__main__':
    from torchstat import stat

    net = STN_ResNet(ResidulBlock=BasicBlock, in_channels=3, num_classes=10)
    stat(net, (3, 32, 32))