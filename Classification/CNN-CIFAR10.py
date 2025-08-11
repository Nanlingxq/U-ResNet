import torch
import torch.utils.data as Data
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
from torchvision.models import vit_b_16
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
import matplotlib.pyplot as plt
from modules import *
import pandas as pd
from resnest_CIFAR10 import resnest

Epochs = 100
batch_size = 64
num_workers = 0
valid_size = 0.2


train_result = []

train_on_gpu = torch.cuda.is_available()
if train_on_gpu:
    print("Cuda is available, The training will be processed on GPU")
device = torch.device("cuda" if train_on_gpu else "cpu")

def ModelEval(criterion, model, test_loader, Epoch, train_loss, train_on_gpu = train_on_gpu):
    test_loss = 0.0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        test_loss += loss.item() * data.size(0)
        _, pred = torch.max(output, 1)
        correct_tensor = pred.eq(target.data.view_as(pred))
        correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())

        for i in range(len(target)):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    test_loss = test_loss / len(test_loader.dataset)
    print('Test Loss: {:.6f}'.format(test_loss))

    insert = {"Epoch":Epoch, "Test Loss":test_loss, "Train Loss":train_loss}

    for i in range(10):
        if class_total[i] > 0:
            insert[classes[i]] = 100 * class_correct[i] / class_total[i]

        else:
            print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

    accuracy = 100. * np.sum(class_correct) / np.sum(class_total)
    # print('Test Accuracy (Overall): %2d%% (%2d/%2d)' % (
    #     100. * np.sum(class_correct) / np.sum(class_total),
    #     np.sum(class_correct), np.sum(class_total)))

    insert['overall'] = accuracy

    train_result.append(insert)

    return accuracy


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

train_data = datasets.CIFAR10('./CIFAR10', train=True, download=True, transform=transform)
test_data = datasets.CIFAR10('./CIFAR10', train=False, download=True, transform=transform)

num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

# define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

# prepare data loaders (combine dataset and sampler)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                           sampler=train_sampler, num_workers=num_workers)
valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                           sampler=valid_sampler, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                          num_workers=num_workers)
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# model = LeNet.LeNet(in_channel=3, features=10).to(device)
# model = Block.BottleCNN(in_channel=3, features=10).to(device)
# model = Block.AlexNet(in_channels=3, features=10).to(device)
# model = Block.CNN(inchannels=3, features=10).to(device)
# model = ResNet.ResNet18(ResNet.BasicBlock, num_classes=10).to(device)
# model = ResNet.ResNet18UpSample(ResNet.BasicBlock, num_classes=10, scale_factor=5).to(device)
# model = DenseNet.DenseNetUpSample(init_channels=64, num_classes=10).to(device)
# model = GoogleNet.GoogLeNet(num_classes=10, aux_logits=False).to(device)
# model = STN_ResNet.STN_ResNet(ResidulBlock=STN_ResNet.BasicBlock, in_channels=3, num_classes=10).to(device)
# model = ViT.ViT(image_size=32, patch_size=8, num_classes=10, dim=1024, depth=6, heads=16, mlp_dim=2048, dropout=0.1, emb_dropout=0.1).to(device)
# model = MobileNet.MobleNetV1(num_classes=10).to(device)
# model = NlNet.NlNet(num_classes=10, in_channels=3).to(device)
# model = resnest.resnet50(num_classes=10).to(device)
# model = EffecientNetv2_CIFAR.efficientnetv2_s(num_classes=10).to(device)
# model = RMT.RMT_S().to(device)
# model1 = NlNet_CIFAR.U_ResNet().to(device)
# model2 = NlNet_CIFAR.E_ResNet().to(device)
# model3 = NlNet_CIFAR.S_ResNet().to(device)
# model4 =NlNet_CIFAR.ResNet18().to(device)
# model1 = NlNet_CIFAR.UResNet3_No_EUCB(num_classes=10, in_channels=3, scale=3).to(device)
# model2 = NlNet_CIFAR.UResNet3_No_UBlock(num_classes=10, in_channels=3, scale=3).to(device)
# model = EffecientNetv2.efficientnetv2_m(num_classes=10).to(device)
model = RegNet.create_regnet(model_name="regnetx_3.2gf", num_classes=10).to(device)
model = NlNet_CIFAR.UResNet4_No_SU(num_classes=10, in_channels=3, scale=3).to(device)
# print("——————————————————————————————————模型架构————————————————————————————————")
# print(model)
# print("——————————————————————————————————模型架构————————————————————————————————")

import torch.optim as optim
def Train(model):
    initial_learning_rate = 0.01
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=initial_learning_rate)


    valid_loss_min = np.inf
    accuracyList = []
    stayLearn = 0

    for epoch in range(1, Epochs + 1):
        train_loss = 0
        valid_loss = 0

        model.train()
        for data, target in train_loader:

            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)

        model.eval()
        for data, target in valid_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            valid_loss += loss.item() * data.size(0)

        train_loss = train_loss / len(train_loader.sampler)
        valid_loss = valid_loss / len(valid_loader.sampler)

        print('Epoch: {}/{} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, Epochs, train_loss, valid_loss))

        accuracy = ModelEval(criterion, model, test_loader, train_loss, epoch)
        accuracyList.append(accuracy)


        if len(accuracyList) >= 10:
            if abs(sum(accuracyList[-10:]) / 10 - accuracy) < 3:
                stayLearn += 1
            else:
                stayLearn = 0
            if stayLearn >= 15:
                initial_learning_rate *= 0.1
                for param_group in optimizer.param_groups:
                    param_group['lr'] = initial_learning_rate
                stayLearn = 0

        print(f"accuracy = {accuracy}%, learningRate = {initial_learning_rate}")

        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, valid_loss))
            torch.save(model.state_dict(), f'./CIFAR10_Result/model_cifar_{model.name}.pt')
            valid_loss_min = valid_loss

        pd.DataFrame(train_result).to_excel(f'./CIFAR10_Result/{model.name}-CIFAR10-TrainResult.xlsx', index=False)

# Train(model1)
# Train(model2)
# Train(model3)
# Train(model4)
# Train(model1)
# Train(model2)
Train(model)