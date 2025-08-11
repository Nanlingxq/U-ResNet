import os
import shutil
import torch
import torch.nn as nn
from IceSeg import IceSegDataset
from sklearn.model_selection import KFold  # 改为使用KFold
from torchvision import transforms as transforms
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np
from modules import *
import torch.optim as optim
from torch.utils.data import DataLoader, Subset  # 添加Subset
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, jaccard_score
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

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
batch_size = 8
n_splits = 10  # 10折交叉验证

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
full_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
print(f"Total samples: {len(full_dataset)}")

# 创建KFold对象
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# 存储每折的结果
all_fold_results = []

# 设备设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 进行10折交叉验证
for fold, (train_idx, val_idx) in enumerate(kfold.split(range(len(full_dataset)))):
    print(f"\n{'='*50}")
    print(f"Fold {fold+1}/{n_splits}")
    print(f"{'='*50}")
    
    # 创建训练集和验证集的子集
    train_subset = Subset(full_dataset, train_idx)
    val_subset = Subset(full_dataset, val_idx)
    
    # 创建数据加载器
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    
    print(f"Training samples: {len(train_subset)}")
    print(f"Validation samples: {len(val_subset)}")
    
    # 初始化模型（每折使用新的模型）
    model = U_ResNet.UResNetNewEUCB(seg_classes=1, in_channels=3, base_c=32).to(device)
    # 也可以尝试其他模型...
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    fold_results = []
    best_val_loss = float('inf')
    
    # 训练模型
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        with tqdm(train_loader, unit="batch", desc=f"Epoch {epoch+1}/{epochs}") as train_batch:
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
        
        # 验证模型
        model.eval()
        val_loss = 0.0
        all_true_masks = []
        all_pred_masks = []
        
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)
                pred_masks = (torch.sigmoid(outputs) > 0.5).float().cpu().numpy()
                true_masks = masks.cpu().numpy()
                all_true_masks.extend(true_masks)
                all_pred_masks.extend(pred_masks)
        
        val_loss /= len(val_loader.dataset)
        all_true_masks = np.array(all_true_masks)
        all_pred_masks = np.array(all_pred_masks)
        
        metrics = compute_segmentation_metrics(all_true_masks, all_pred_masks)
        metrics['val_loss'] = val_loss
        metrics['train_loss'] = epoch_loss
        metrics['epoch'] = epoch + 1
        metrics['fold'] = fold + 1
        
        fold_results.append(metrics)
        
        print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {val_loss:.4f}")
        print(f"Metrics - Acc: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}, IoU: {metrics['iou']:.4f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'Result/{model.name}_fold{fold+1}_best_weights.pth')
            print(f"Fold {fold+1} best model saved with val_loss: {val_loss:.4f}")
    
    # 保存当前折的结果
    fold_df = pd.DataFrame(fold_results)
    fold_df.to_excel(f'Result/{model.name}_fold{fold+1}_results.xlsx', index=False)
    all_fold_results.append(fold_df)
    
    print(f"Fold {fold+1} completed. Best validation loss: {best_val_loss:.4f}")

# 合并所有折的结果并保存
final_results = pd.concat(all_fold_results)
final_results.to_excel(f'Result/{model.name}_all_folds_results.xlsx', index=False)

# 计算平均性能指标
mean_metrics = final_results.groupby('epoch').mean(numeric_only=True)
mean_metrics.to_excel(f'Result/{model.name}_mean_metrics.xlsx')

print("10-fold cross validation completed!")

