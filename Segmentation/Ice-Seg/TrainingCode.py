import os
import shutil
import torch
import torch.nn as nn
from IceSeg import IceSegDataset
from sklearn.model_selection import train_test_split
from torchvision import transforms as transforms
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np
from modules import *
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, jaccard_score

def compute_segmentation_metrics(true_masks, pred_masks):
    """
    计算分割任务的各种指标。

    参数：
        true_masks (numpy.ndarray): 真实掩码，形状为 (num_samples, height, width)
        pred_masks (numpy.ndarray): 预测掩码，形状为 (num_samples, height, width)

    返回：
        dict: 包含各种指标的字典
    """
    # 将掩码展平为一维数组
    true_masks_flat = true_masks.flatten()
    pred_masks_flat = pred_masks.flatten()

    # 计算精确率、召回率、F1 分数，设置 zero_division 参数
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_masks_flat, pred_masks_flat, average='binary', zero_division=0
    )

    # 计算 IoU，设置 zero_division 参数
    iou = jaccard_score(true_masks_flat, pred_masks_flat, zero_division=0)

    # 计算准确率
    accuracy = (true_masks_flat == pred_masks_flat).mean()

    # 计算 Dice 系数，添加一个小值避免分母为零
    intersection = np.sum(true_masks_flat * pred_masks_flat)
    dice = (2. * intersection) / (np.sum(true_masks_flat) + np.sum(pred_masks_flat) + 1e-6)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'iou': iou,
        'dice_coefficient': dice
    }

learning_rates = [0.1, 0.01, 0.001]
epochs = 50
batch_size = 2

image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

mask_transform = transforms.Compose([
    transforms.ToTensor(),
])

# 创建数据集实例
train_dataset = IceSegDataset(
    yaml_path='IceSeg/data.yaml',
    dataset_type='train',
    image_transforms=image_transform,
    mask_transforms = mask_transform
)
test_dataset = IceSegDataset(
    yaml_path='IceSeg/data.yaml',
    dataset_type='test',
    image_transforms=image_transform,
    mask_transforms=mask_transform
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print("Data load successfully!")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = U_ResNet.UResNet(seg_classes=1, in_channels=3, base_c=64).to(device)
# model = RMT.RMT_Segmentation(
#     num_classes=1, in_chans=3, out_indices=(0, 1, 2, 3),
#     embed_dims=[96, 192, 384, 768], depths=[3, 6, 15, 6], num_heads=[4, 4, 8, 8],
#     init_values=[2, 2, 2, 2], heads_ranges=[6, 6, 6, 6], mlp_ratios=[4, 4, 3, 3], drop_path_rate=0.1
# ).to(device)
# model = ResUnetPP.build_resunetplusplus().to(device)
# model = U_ResNet.UNet().to(device)
# model = U_ResNet.UResNetNoEUCB().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
List = []

best_val_loss = float('inf')
# 训练模型
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    with tqdm(train_loader, unit="batch") as train_batch:
        for images, masks in train_batch:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            train_batch.set_postfix(loss=loss.item())

    epoch_loss = train_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

    all_true_masks = []
    all_pred_masks = []

    # 验证模型
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item() * images.size(0)
            pred_masks = (torch.sigmoid(outputs) > 0.5).float().cpu().numpy()
            true_masks = masks.cpu().numpy()
            all_true_masks.extend(true_masks)
            all_pred_masks.extend(pred_masks)

    val_loss /= len(test_loader.dataset)
    all_true_masks = np.array(all_true_masks)
    all_pred_masks = np.array(all_pred_masks)
    metrics = compute_segmentation_metrics(all_true_masks, all_pred_masks)
    metrics['testLoss'] = val_loss
    metrics['trainLoss'] = epoch_loss
    List.append(metrics)
    pd.DataFrame(List).to_excel(f'Result/{model.name}(64)_result.xlsx', index=False)
    print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {val_loss:.4f} ")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), f'Result/{model.name}(64)_best_weights.pth')
        print("模型参数保存成功")

