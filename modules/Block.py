import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib

class CNN(nn.Module):
    def __init__(self, inchannels, features):
        super(CNN, self).__init__()
        self.name = 'CNN'
        self.conv1 = nn.Conv2d(inchannels, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 4 * 4, 500)
        self.fc2 = nn.Linear(500, features)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(-1, 64 * 4 * 4)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

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

class STNBlock(nn.Module):
    def __init__(self, traget_features):
        super(STN, self).__init__()
        self.name = 'Spatial Transformer Networks'
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=5, padding=1)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(32 * 4, 256)
        self.fc2 = nn.Linear(256, traget_features) #输出的目标数量


        self.localization = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, padding=2),
            nn.MaxPool2d(3, stride=2),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=5),
            nn.MaxPool2d(3, stride=2),
            nn.ReLU(True)
        )

        self.fc_loc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def stn(self, x):
        xs = self.localization(x)
        # print(f"xs_size1 =  {xs.size()}")
        xs = xs.view(-1,  16 * 4)
        # print(f"xs_size2 =  {xs.size()}")
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        print(f"theta_size =  {theta.size()}")
        print(f"x_size =  {x.size()}")
        x = x.view(-1, 3, 16, 16)

        grid = F.affine_grid(theta=theta, size=x.size(), align_corners=True)
        x = F.grid_sample(x, grid, align_corners=True)

        return x

    def forward(self, x):
        x = self.stn(x)
        # print(f"x.size = {x.size()}")
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # print(f"x1.size() = {x.size()}")
        x = x.view(-1, 32 * 8)
        # print(f"x2.size() = {x.size()}")
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)

class BottleCNN(nn.Module):
    def __init__(self, in_channel, features):
        super(BottleCNN, self).__init__()
        self.name = 'BottleCNN'
        self.conv1 = nn.Conv2d(in_channel, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.BN = Bottleneck(32, 64, 1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 4 * 4, 500)
        self.fc2 = nn.Linear(500, features)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.BN(x)))

        x = x.view(-1, 64 * 4 * 4)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class LeNet(nn.Module):
    def __init__(self, in_channel, features):
        super(LeNet, self).__init__()
        self.name = "LeNet"
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=6, kernel_size=5, stride=1, padding=0)
        self.avg_pooling1 = nn.AvgPool2d(kernel_size=7, stride=3)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.avg_pooling2 = nn.AvgPool2d(kernel_size=7, stride=3)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=5, stride=1, padding=0)
        self.avg_pooling3 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc_layer1 = nn.Linear(in_features=1024 * 4, out_features=84)
        self.fc_layer2 = nn.Linear(in_features=84, out_features=features)
        self.drop = nn.Dropout(0.4)

    def forward(self, x):
        # print(f"x0.size() = {x.size()}")
        x = F.relu(self.conv1(x))
        # print(f"x1.size() = {x.size()}")
        x = self.avg_pooling1(x)
        # print(f"x2.size() = {x.size()}")
        x = F.relu(self.conv2(x))
        # print(f"x3.size() = {x.size()}")
        x = self.avg_pooling2(x)
        # print(f"x4.size() = {x.size()}")
        x = F.relu(self.conv3(x))
        x = self.avg_pooling3(x)
        x = self.flatten(x)
        x = self.drop(x)
        x = F.relu(self.fc_layer1(x))
        x = self.drop(x)
        # print(f"x5.size() = {x.size()}")
        x = self.fc_layer2(x)
        # print(f"x6.size() = {x.size()}")

        return x

class AlexNet(nn.Module):
    def __init__(self, in_channels, features):
        super(AlexNet, self).__init__()
        self.name = "AlexNet"
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 48, kernel_size=11),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(48, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(128, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048, features)
        )

        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        # print(x.size())
        x = self.classifier(x)

        return x