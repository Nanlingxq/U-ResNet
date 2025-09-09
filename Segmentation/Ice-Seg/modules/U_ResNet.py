import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x

def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'relu6':
        layer = nn.ReLU6(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer

class EUCB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, activation='relu', scale_factor=2):
        super(EUCB, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up_dwc = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode="bilinear"),
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2, groups=self.in_channels, bias=False),
            nn.BatchNorm2d(self.in_channels),
            act_layer(activation, inplace=True)
        )
        self.pwc = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x):
        x = self.up_dwc(x)
        x = channel_shuffle(x, self.in_channels)
        x = self.pwc(x)
        return x

class EUCB2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, activation='relu', scale_factor=2):
        super(EUCB2, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up_dwc1 = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode="bilinear"),
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2, groups=self.in_channels // 4, bias=False),
            nn.BatchNorm2d(self.in_channels),
            act_layer(activation, inplace=True)
        )
        self.up_dwc2 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2, groups=self.in_channels // 4, bias=False),
            nn.BatchNorm2d(self.in_channels),
            act_layer(activation, inplace=True)
        )
        self.pwc = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x):
        x = self.up_dwc1(x)
        x = channel_shuffle(x, self.in_channels)
        x = self.up_dwc2(x)
        x = self.pwc(x)
        return x

class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConv(in_channels, out_channels)
        )

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()

        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)

        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x
class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )

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
class UNet(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 num_classes: int = 1,
                 base_c: int = 64):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        self.down4 = Down(base_c * 8, base_c * 16)

        self.up1 = Up(base_c * 16, base_c * 8)
        self.up2 = Up(base_c * 8, base_c * 4)
        self.up3 = Up(base_c * 4, base_c * 2)
        self.up4 = Up(base_c * 2, base_c)
        self.out_conv = OutConv(base_c, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.out_conv(x)

        return logits
class UResNet(nn.Module):
    def __init__(self, seg_classes=1, in_channels=3, base_c=32):
        super(UResNet, self).__init__()
        self.name = "U-ResNet"

        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.ResLayer1 = self.make_layer(BasicBlock, 64, 2, stride=1)
        self.ResLayer2 = self.make_layer(BasicBlock, 128, 2, stride=1)
        self.ResLayer3 = self.make_layer(BasicBlock, 64, 2, stride=1)
        self.ResLayer4 = self.make_layer(BasicBlock, 32, 2, stride=1)
        self.Upsample = EUCB(in_channels=32, out_channels=16, scale_factor=2)

        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        self.down4 = Down(base_c * 8, base_c * 16)
        self.up1 = Up(base_c * 16, base_c * 8)
        self.up2 = Up(base_c * 8, base_c * 4)
        self.up3 = Up(base_c * 4, base_c * 2)
        self.up4 = Up(base_c * 2, base_c)

        self.out_conv = OutConv(base_c + 16, seg_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self,x):
        # ResNet processing
        Res_out = self.conv1(x)
        Res_out = self.ResLayer1(Res_out)
        Res_out = self.ResLayer2(Res_out)
        Res_out = self.ResLayer3(Res_out)
        Res_out = self.ResLayer4(Res_out)
        Res_out = self.Upsample(Res_out)

        # U-Net processing
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        U_out = self.up1(x5, x4)
        U_out = self.up2(U_out, x3)
        U_out = self.up3(U_out, x2)
        U_out = self.up4(U_out, x1)

        out = torch.cat([U_out, Res_out], dim=1)

        out = self.out_conv(out)

        return out
class UResNetNewEUCB(nn.Module):
    def __init__(self, seg_classes=1, in_channels=3, base_c=32):
        super(UResNetNewEUCB, self).__init__()
        self.name = "U-ResNetNewEUCB"

        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.ResLayer1 = self.make_layer(BasicBlock, 64, 2, stride=1)
        self.ResLayer2 = self.make_layer(BasicBlock, 128, 2, stride=1)
        self.ResLayer3 = self.make_layer(BasicBlock, 64, 2, stride=1)
        self.ResLayer4 = self.make_layer(BasicBlock, 32, 2, stride=1)
        self.Upsample = EUCB2(in_channels=32, out_channels=16, scale_factor=2)

        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        self.down4 = Down(base_c * 8, base_c * 16)
        self.up1 = Up(base_c * 16, base_c * 8)
        self.up2 = Up(base_c * 8, base_c * 4)
        self.up3 = Up(base_c * 4, base_c * 2)
        self.up4 = Up(base_c * 2, base_c)

        self.out_conv = OutConv(base_c + 16, seg_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self,x):
        # ResNet processing
        Res_out = self.conv1(x)
        Res_out = self.ResLayer1(Res_out)
        Res_out = self.ResLayer2(Res_out)
        Res_out = self.ResLayer3(Res_out)
        Res_out = self.ResLayer4(Res_out)
        Res_out = self.Upsample(Res_out)

        # U-Net processing
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        U_out = self.up1(x5, x4)
        U_out = self.up2(U_out, x3)
        U_out = self.up3(U_out, x2)
        U_out = self.up4(U_out, x1)

        out = torch.cat([U_out, Res_out], dim=1)

        out = self.out_conv(out)

        return out
class UResNetNoEUCB(nn.Module):
    def __init__(self, seg_classes=1, in_channels=3, base_c=32):
        super(UResNetNoEUCB, self).__init__()
        self.name = "U-ResNet-NoEUCB"

        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.ResLayer1 = self.make_layer(BasicBlock, 64, 2, stride=1)
        self.ResLayer2 = self.make_layer(BasicBlock, 128, 2, stride=1)
        self.ResLayer3 = self.make_layer(BasicBlock, 64, 2, stride=1)
        self.ResLayer4 = self.make_layer(BasicBlock, 16, 2, stride=1)
        self.Upsample = nn.Upsample(scale_factor=2, mode="bilinear")

        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        self.down4 = Down(base_c * 8, base_c * 16)
        self.up1 = Up(base_c * 16, base_c * 8)
        self.up2 = Up(base_c * 8, base_c * 4)
        self.up3 = Up(base_c * 4, base_c * 2)
        self.up4 = Up(base_c * 2, base_c)

        self.out_conv = OutConv(base_c + 16, seg_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self,x):
        # ResNet processing
        Res_out = self.conv1(x)
        Res_out = self.ResLayer1(Res_out)
        Res_out = self.ResLayer2(Res_out)
        Res_out = self.ResLayer3(Res_out)
        Res_out = self.ResLayer4(Res_out)
        Res_out = self.Upsample(Res_out)

        # U-Net processing
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        U_out = self.up1(x5, x4)
        U_out = self.up2(U_out, x3)
        U_out = self.up3(U_out, x2)
        U_out = self.up4(U_out, x1)

        out = torch.cat([U_out, Res_out], dim=1)

        out = self.out_conv(out)

        return out
class UResNetNoResBlock(nn.Module):
    def __init__(self, seg_classes=1, in_channels=3, base_c=32):
        super(UResNetNoResBlock, self).__init__()
        self.name = "UResNetNoResBlock"

        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        self.down4 = Down(base_c * 8, base_c * 16)
        self.up1 = Up(base_c * 16, base_c * 8)
        self.up2 = Up(base_c * 8, base_c * 4)
        self.up3 = Up(base_c * 4, base_c * 2)
        self.up4 = Up(base_c * 2, base_c)

        self.out_conv = OutConv(base_c, seg_classes)


    def forward(self,x):
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        U_out = self.up1(x5, x4)
        U_out = self.up2(U_out, x3)
        U_out = self.up3(U_out, x2)
        U_out = self.up4(U_out, x1)

        out = self.out_conv(U_out)

        return out