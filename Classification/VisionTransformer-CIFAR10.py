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
import pandas as pd

Epochs = 100
batch_size = 64
num_workers = 0
valid_size = 0.2
initial_learning_rate = 0.001  # ViT 使用较小的学习率

train_result = []

train_on_gpu = torch.cuda.is_available()
if train_on_gpu:
    print("Cuda is available, The training will be processed on GPU")
device = torch.device("cuda" if train_on_gpu else "cpu")

def ModelEval(model, test_loader, Epoch, train_on_gpu=train_on_gpu):
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

    insert = {"Epoch": Epoch, "Test Loss": test_loss, "Train Loss": train_loss}

    for i in range(10):
        if class_total[i] > 0:
            insert[classes[i]] = 100 * class_correct[i] / class_total[i]
        else:
            print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

    accuracy = 100. * np.sum(class_correct) / np.sum(class_total)
    insert['overall'] = accuracy

    train_result.append(insert)

    return accuracy


# 修改数据预处理部分
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])  # ViT 的归一化参数
])

train_data = datasets.CIFAR10('./CIFAR10', train=True, download=True, transform=transform)
test_data = datasets.CIFAR10('./CIFAR10', train=False, download=True, transform=transform)

num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                           sampler=train_sampler, num_workers=num_workers)
valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                           sampler=valid_sampler, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                          num_workers=num_workers)
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# 修改模型定义部分
from torchvision.models import vit_b_16


model = vit_b_16(image_size=32, num_classes=10).to(device)

criterion = nn.CrossEntropyLoss()

# 修改优化器部分
optimizer = torch.optim.AdamW(model.parameters(), lr=initial_learning_rate)

valid_loss_min = np.Inf
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

    accuracy = ModelEval(model, test_loader, epoch)
    accuracyList.append(accuracy)


    if len(accuracyList) >= 10:
        if abs(sum(accuracyList[-10:]) / 10 - accuracy) < 3:
            stayLearn += 1
        else:
            stayLearn = 0
        if stayLearn >= 15:
            initial_learning_rate *= 0.1
            stayLearn = 0

    print(f"accuracy = {accuracy}%, learningRate = {initial_learning_rate}")

    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, valid_loss))
        torch.save(model.state_dict(), f'./CIFAR10_Result/model_cifar_Vit_16b.pt')
        valid_loss_min = valid_loss


    pd.DataFrame(train_result).to_excel(f'./CIFAR10_Result/vit_b16-CIFAR10-TrainResult.xlsx', index=False)