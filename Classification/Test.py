import torch
import torch.utils.data as Data
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
import matplotlib.pyplot as plt
from modules import *
import pandas as pd
from tqdm import tqdm

Epochs = 100
batch_size = 64
num_workers = 0
valid_size = 0.2

train_result = []

train_on_gpu = torch.cuda.is_available()
if train_on_gpu:
    print("Cuda is available, The training will be processed on GPU")
device = torch.device("cuda" if train_on_gpu else "cpu")


def ModelEval(criterion, model, test_loader, Epoch, train_loss, train_on_gpu=train_on_gpu, test_bar=None):
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

        accuracy = len(class_correct) / len(class_total)

        test_bar.set_postfix({
            'test_loss': f"{loss.item():.4f}",
            'accuracy': f"{100 * accuracy:.2f}%",
        })
    test_loss = test_loss / len(test_loader.dataset)
    print('Test Loss: {:.6f}'.format(test_loss))

    insert = {"Epoch": Epoch, "Test Loss": test_loss, "Train Loss": train_loss}

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



# prepare data loaders (combine dataset and sampler)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                          num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                          num_workers=num_workers)
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

model = UResNet.UResNet(10, 3, scale=3).to(device)

import torch.optim as optim


def Train(model):
    initial_learning_rate = 0.001
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    valid_loss_min = np.inf
    accuracyList = []
    stayLearn = 0

    for epoch in range(1, Epochs + 1):
        train_loss = 0
        valid_loss = 0

        model.train()
        train_bar = tqdm(train_loader,
                         desc=f'Epoch {epoch}/{Epochs} [Train]',
                         unit='batch',
                         ncols=100)  # 控制进度条宽度
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)
            train_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': optimizer.param_groups[0]['lr'],
            })

        model.eval()
        test_bar = tqdm(test_loader,
                        desc=f'Epoch {epoch}/{epochs} [Test]',
                        unit='batch',
                        ncols=100,
                        colour='green')
        accuracy = ModelEval(criterion, model, test_loader, train_loss, epoch, test_bar)
        accuracyList.append(accuracy)

        if len(accuracyList) >= 10:
            if abs(sum(accuracyList[-10:]) / 10 - accuracy) < 3:
                stayLearn += 1
            else:
                stayLearn = 0
            if stayLearn >= 15:
                initial_learning_rate *= 0.1
                stayLearn = 0

        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, valid_loss))
            torch.save(model.state_dict(), f'./CIFAR10_Result/model_cifar_{model.name}.pt')
            valid_loss_min = valid_loss

        pd.DataFrame(train_result).to_excel(f'./CIFAR10_Result/{model.name}-CIFAR10-TrainResult.xlsx', index=False)


Train(model)