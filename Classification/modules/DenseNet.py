import torch
import torch.nn as nn
import torch.functional as F


def Conv1(in_channels, places, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_channels, places, kernel_size=7, stride=stride, padding=7, bias=False),
        nn.BatchNorm2d(places),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )

class _DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, bn_size, drop_rate=0):
        super(_DenseLayer,self).__init__()
        self.drop_rate = drop_rate
        self.dense_layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, bn_size * growth_rate, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(bn_size * growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.dropout = nn.Dropout(self.drop_rate)

    def forward(self, x):
        y = self.dense_layer(x)
        if self.drop_rate > 0:
            y = self.dropout(y)
        return torch.concat([x,y], 1)

class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate, bn_size, drop_rate=0):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(_DenseLayer(in_channels + i * growth_rate, growth_rate, bn_size, drop_rate))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class _TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(_TransitionLayer, self).__init__()
        self.transition_layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=1),
        )

    def forward(self, x):
        return self.transition_layer(x)

class DenseNet(nn.Module):
    def __init__(self, init_channels=64, growth_rate=32, blocks=[6, 12, 24, 16], num_classes=10):
        super(DenseNet, self).__init__()
        self.name = "DenseNet"
        bn_size = 4
        drop_rate = 0
        self.conv1 = Conv1(3, init_channels)

        num_features = init_channels

        # 第1个DenseBlock有6个DenseLayer, 执行DenseBlock（6,64,32,4）
        self.layer1 = DenseBlock(num_layers=blocks[0], in_channels=num_features, growth_rate=growth_rate, bn_size=bn_size,
                                 drop_rate=drop_rate)
        num_features = num_features + blocks[0] * growth_rate
        # 第1个transition 执行 _TransitionLayer（256,128）
        self.transition1 = _TransitionLayer(in_channels=num_features, out_channels=num_features // 2)
        # num_features减少为原来的一半，执行第1回合之后，第2个DenseBlock的输入的feature应该是：num_features = 128
        num_features = num_features // 2

        # 第2个DenseBlock有12个DenseLayer, 执行DenseBlock（12,128,32,4）
        self.layer2 = DenseBlock(num_layers=blocks[1], in_channels=num_features, growth_rate=growth_rate, bn_size=bn_size,
                                 drop_rate=drop_rate)
        num_features = num_features + blocks[1] * growth_rate
        # 第2个transition 执行 _TransitionLayer（512,256）
        self.transition2 = _TransitionLayer(in_channels=num_features, out_channels=num_features // 2)
        # num_features减少为原来的一半，执行第2回合之后，第3个DenseBlock的输入的feature应该是：num_features = 256
        num_features = num_features // 2

        # 第3个DenseBlock有24个DenseLayer, 执行DenseBlock（24,256,32,4）
        self.layer3 = DenseBlock(num_layers=blocks[2], in_channels=num_features, growth_rate=growth_rate, bn_size=bn_size,
                                 drop_rate=drop_rate)
        num_features = num_features + blocks[2] * growth_rate
        # 第3个transition 执行 _TransitionLayer（1024,512）
        self.transition3 = _TransitionLayer(in_channels=num_features, out_channels=num_features // 2)
        # num_features减少为原来的一半，执行第3回合之后，第4个DenseBlock的输入的feature应该是：num_features = 512
        num_features = num_features // 2

        # 第4个DenseBlock有16个DenseLayer, 执行DenseBlock（16,512,32,4）
        self.layer4 = DenseBlock(num_layers=blocks[3], in_channels=num_features, growth_rate=growth_rate, bn_size=bn_size,
                                 drop_rate=drop_rate)
        num_features = num_features + blocks[3] * growth_rate

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.transition1(x)
        # print(f"x1.size() = {x.size()}")
        x = self.layer2(x)
        x = self.transition2(x)
        # print(f"x2.size() = {x.size()}")
        x = self.layer3(x)
        x = self.transition3(x)
        x = self.layer4(x)
        # print(f"x3.size() = {x.size()}")

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # print(f"x4.size() = {x.size()}")
        x = self.fc(x)
        return x


class DenseNetUpSample(nn.Module):
    def __init__(self, init_channels=64, growth_rate=32, blocks=[6, 12, 24, 16], num_classes=10, scale_feature=3):
        super(DenseNetUpSample, self).__init__()
        self.name = f"DenseNetUp-scale-{scale_feature}"
        bn_size = 4
        drop_rate = 0
        self.conv1 = Conv1(3, init_channels)
        self.Upsample = nn.Upsample(scale_factor=scale_feature, mode="bilinear")
        num_features = init_channels

        # 第1个DenseBlock有6个DenseLayer, 执行DenseBlock（6,64,32,4）
        self.layer1 = DenseBlock(num_layers=blocks[0], in_channels=num_features, growth_rate=growth_rate,
                                 bn_size=bn_size,
                                 drop_rate=drop_rate)
        num_features = num_features + blocks[0] * growth_rate
        # 第1个transition 执行 _TransitionLayer（256,128）
        self.transition1 = _TransitionLayer(in_channels=num_features, out_channels=num_features // 2)
        # num_features减少为原来的一半，执行第1回合之后，第2个DenseBlock的输入的feature应该是：num_features = 128
        num_features = num_features // 2

        # 第2个DenseBlock有12个DenseLayer, 执行DenseBlock（12,128,32,4）
        self.layer2 = DenseBlock(num_layers=blocks[1], in_channels=num_features, growth_rate=growth_rate,
                                 bn_size=bn_size,
                                 drop_rate=drop_rate)
        num_features = num_features + blocks[1] * growth_rate
        # 第2个transition 执行 _TransitionLayer（512,256）
        self.transition2 = _TransitionLayer(in_channels=num_features, out_channels=num_features // 2)
        # num_features减少为原来的一半，执行第2回合之后，第3个DenseBlock的输入的feature应该是：num_features = 256
        num_features = num_features // 2

        # 第3个DenseBlock有24个DenseLayer, 执行DenseBlock（24,256,32,4）
        self.layer3 = DenseBlock(num_layers=blocks[2], in_channels=num_features, growth_rate=growth_rate,
                                 bn_size=bn_size,
                                 drop_rate=drop_rate)
        num_features = num_features + blocks[2] * growth_rate
        # 第3个transition 执行 _TransitionLayer（1024,512）
        self.transition3 = _TransitionLayer(in_channels=num_features, out_channels=num_features // 2)
        # num_features减少为原来的一半，执行第3回合之后，第4个DenseBlock的输入的feature应该是：num_features = 512
        num_features = num_features // 2

        # 第4个DenseBlock有16个DenseLayer, 执行DenseBlock（16,512,32,4）
        self.layer4 = DenseBlock(num_layers=blocks[3], in_channels=num_features, growth_rate=growth_rate,
                                 bn_size=bn_size,
                                 drop_rate=drop_rate)
        num_features = num_features + blocks[3] * growth_rate

        self.avgpool = nn.AvgPool2d(7, stride=4)
        self.fc = nn.Linear(25600, num_classes)

    def forward(self, x):
        x = self.Upsample(x)
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.transition1(x)
        # print(f"x1.size() = {x.size()}")
        x = self.layer2(x)
        x = self.transition2(x)
        # print(f"x2.size() = {x.size()}")
        x = self.layer3(x)
        x = self.transition3(x)
        x = self.layer4(x)
        # print(f"x3.size() = {x.size()}")

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # print(f"x4.size() = {x.size()}")
        x = self.fc(x)
        return x