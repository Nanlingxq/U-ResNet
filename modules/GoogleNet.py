import torch
from torch import nn
from torch.nn import functional as F

class GoogLeNet(nn.Module):
    def __init__(self, num_classes=10, aux_logits=False):
        super().__init__()
        self.name = "GoogleNet"
        self.aux_logits = aux_logits

        # self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.conv1 = BasicConv2d(3, 64, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.conv2 = BasicConv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        if self.aux_logits:
            self.aux1 = AuxClassifier(512, num_classes)
            self.aux2 = AuxClassifier(528, num_classes)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, X):
        X = self.conv1(X)
        # X = self.maxpool1(X)
        X = self.conv2(X)
        # X = self.maxpool2(X)

        X = self.inception3a(X)
        X = self.inception3b(X)
        X = self.maxpool3(X)

        X = self.inception4a(X)
        if self.training and self.aux_logits:
            aux1 = self.aux1(X)
        X = self.inception4b(X)
        X = self.inception4c(X)
        X = self.inception4d(X)
        if self.training and self.aux_logits:
            aux2 = self.aux2(X)
        X = self.inception4e(X)
        X = self.maxpool4(X)

        X = self.inception5a(X)
        X = self.inception5b(X)

        X = self.avgpool(X)
        X = torch.flatten(X, start_dim=1)
        X = self.dropout(X)
        X = self.fc(X)

        if self.training and self.aux_logits:
            return X, aux2, aux1
        return X


class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super().__init__()

        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=5, padding=2)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, X):
        branch1 = self.branch1(X)
        branch2 = self.branch2(X)
        branch3 = self.branch3(X)
        branch4 = self.branch4(X)

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, dim=1)


class AuxClassifier(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.averagePool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = BasicConv2d(in_channels, 128, kernel_size=1)

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, X):
        X = self.averagePool(X)
        X = self.conv(X)

        X = torch.flatten(X, start_dim=1)

        X = F.relu(self.fc1(X), inplace=True)
        X = F.dropout(X, 0.7, training=self.training)
        X = self.fc2(X)

        return X


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X):
        return self.relu(self.bn(self.conv(X)))


if __name__ == '__main__':
    from torchstat import stat

    net = GoogLeNet()
    stat(net, (3, 32, 32))
