import os
import shutil
import torch
import torch.nn as nn
from DataInstance import *
from sklearn.model_selection import train_test_split
from torchvision import transforms as transforms
from modules import *
import pandas as pd


def ReSetDataSet(test_size = 0.2, data_dir=r"miniImageNet", train_dir=r"miniImageNet_d/train", test_dir=r"miniImageNet_d/test"):

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for class_name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            image_files = [f for f in os.listdir(class_dir) if f.endswith(('JPEG'))]
            train_files, test_files = train_test_split(image_files, test_size=test_size, random_state=42)

            os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
            os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

            for file in train_files:
                shutil.copy(os.path.join(class_dir, file), os.path.join(train_dir, class_name, file))
            for file in test_files:
                shutil.copy(os.path.join(class_dir, file), os.path.join(test_dir, class_name, file))

    print(f"数据划分完成，训练集 : 测试集 = {(1 - test_size) * 10} : {test_size * 10}")

learning_rates = [0.1, 0.01, 0.001]
epochs = 100
batch_size = 32

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dir = r"miniImageNet_d/train"
test_dir = r"miniImageNet_d/test"

train_dataset = ImageNet.ImageDataset(train_dir, transform=train_transform)
test_dataset = ImageNet.ImageDataset(test_dir, transform=test_transform)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(train_dataset.classes)
# model = Block.LeNet(in_channel=3, features=100).to(device)
model = ResNet.ResNet18(ResNet.BasicBlock, num_classes=100).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rates[0])

train_list = []

if __name__ == "__main__":
    for epoch in range(1, epochs + 1):
        if epoch >= 0.4 * epochs and epoch <= 0.8 * epochs:
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rates[1])
        elif epoch > 0.8 * epochs:
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rates[2])

        train_loss = 0
        test_loss = 0

        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        model.eval()
        correct_top1 = 0
        correct_top5 = 0
        total = 0
        with torch.no_grad(): #禁用梯度计算，加快运行速度，减少内存开销
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted_top1 = torch.max(outputs.data, 1)
                _, predicted_top5 = torch.topk(outputs.data, 5, dim=1)

                total += labels.size(0)
                correct_top1 += (predicted_top1 == labels).sum().item()
                correct_top5 += torch.sum(predicted_top5.eq(labels.view(-1, 1))).item()
                test_loss += loss.item() * images.size(0)

        train_loss = train_loss / len(train_loader.sampler)
        test_loss = test_loss / len(test_loader.sampler)

        accuracy_top1 = 100 * correct_top1 / total
        accuracy_top5 = 100 * correct_top5 / total
        temp = {"Top1_accuracy":accuracy_top1, "Top5_accuracy":accuracy_top5, "train_loss":train_loss, "test_loss":test_loss}

        train_list.append(temp)

        print(f'Epoch [{epoch}/{epochs}]\t Test Accuracy_top1: {accuracy_top1:.2f}% \t Test Accuracy_top5: {accuracy_top5:.2f}% \t trainLoss:{train_loss}\t testLoss:{test_loss}')
        pd.DataFrame(train_list).to_excel(f'./MiniImageNetResult/{model.name}-MiniImageNet-TrainResult.xlsx', index=False)