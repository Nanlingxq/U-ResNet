import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import time
import os
from modules import *
from DataInstances import Cityscapes
from tqdm import tqdm
from torch.nn.parallel import DataParallel
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, jaccard_score
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd

def get_data_loaders(root_dir, batch_size=2, num_workers=4, IsSubset=False):
    """获取语义分割数据加载器"""
    # 使用语义分割数据集
    train_dataset = Cityscapes.CityscapesSemantic(
        root_dir=root_dir,
        split='train',
        transform=Cityscapes.CityscapesSemantic.get_default_transform()
    )
    
    val_dataset = Cityscapes.CityscapesSemantic(
        root_dir=root_dir,
        split='val',
        transform=Cityscapes.CityscapesSemantic.get_default_transform()
    )
    
    if IsSubset:
        loader = DataLoader(
            Subset(train_dataset, indices=range(100)),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )

    # 语义分割不需要特殊collate_fn
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

def compute_loss(outputs, targets):
    """计算语义分割损失"""
    if isinstance(outputs, list):
        outputs = outputs[0]
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    loss = criterion(outputs, targets.long())
    return loss

def calculate_metrics(preds, targets, num_classes):
    """计算评估指标"""
    # 展平预测和目标
    preds_flat = preds.argmax(1).flatten().cpu().numpy()
    targets_flat = targets.flatten().cpu().numpy()
    
    # 忽略255（忽略类）
    valid_mask = targets_flat != 255
    preds_flat = preds_flat[valid_mask]
    targets_flat = targets_flat[valid_mask]
    
    # 计算指标
    precision = precision_score(targets_flat, preds_flat, average='macro', zero_division=0)
    recall = recall_score(targets_flat, preds_flat, average='macro', zero_division=0)
    accuracy = accuracy_score(targets_flat, preds_flat)
    f1 = f1_score(targets_flat, preds_flat, average='macro', zero_division=0)
    
    # 计算IoU (Jaccard Index)
    iou = jaccard_score(targets_flat, preds_flat, average='macro', zero_division=0)
    
    # 计算Dice系数（F1的别名）
    dice = f1
    
    return {
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,
        'f1': f1,
        'iou': iou,
        'dice': dice
    }

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    num_batches = len(data_loader)
    
    pbar = tqdm(data_loader, total=num_batches, desc=f"Epoch {epoch+1}")
    
    for images, targets in pbar:
        images = images.to(device)
        targets = targets.to(device)
        
        # 语义分割模型直接输出分割图
        outputs = model(images)
        # print(f"outputs.size() = {outputs.size()}")
        loss = compute_loss(outputs, targets)
        total_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        pbar.set_postfix(loss=loss.item())
    
    return total_loss / num_batches

def validate(model, data_loader, device, num_classes):
    """验证模型并计算指标"""
    model.eval()
    total_loss = 0
    num_batches = len(data_loader)
    
    # 初始化指标字典
    metrics = {
        'precision': 0,
        'recall': 0,
        'accuracy': 0,
        'f1': 0,
        'iou': 0,
        'dice': 0
    }
    
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Validating"):
            images = images.to(device)
            targets = targets.to(device)
            
            outputs = model(images)
            loss = compute_loss(outputs, targets)
            total_loss += loss.item()
            
            # 计算指标
            batch_metrics = calculate_metrics(outputs, targets, num_classes)
            
            # 累加指标
            for key in metrics:
                metrics[key] += batch_metrics[key]
    
    # 计算平均指标
    for key in metrics:
        metrics[key] /= num_batches
    
    return total_loss / num_batches, metrics

def main():
    root_dir = 'cityscapes'
    num_classes = 19
    num_epochs = 50
    batch_size = 6
    learning_rate = 0.0001
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = 'Results'
    os.makedirs(save_dir, exist_ok=True)
    
    # 获取数据加载器（直接使用训练集和测试集）
    train_loader, test_loader = get_data_loaders(root_dir, batch_size, num_workers=4)
    
    # 初始化模型
    # model = RMT.RMT_Segmentation(
    #     num_classes=num_classes, in_chans=3, out_indices=(0, 1, 2, 3),
    #     embed_dims=[56, 112, 224, 320], depths=[4, 8, 25, 8], num_heads=[7, 7, 14, 20],
    #     init_values=[2, 2, 2, 2], heads_ranges=[6, 6, 6, 6], mlp_ratios=[4, 4, 3, 3], drop_path_rate=0.5
    # )
    
    # model = U_ResNet.UResNetNewEUCB(seg_classes=num_classes, in_channels=3, base_c=64)

    model = U_ResNet.UResNetNoEUCB(seg_classes=num_classes, in_channels=3, base_c=64)

    # model = U_ResNet.UNet(seg_classes=num_classes, in_channels=3, base_c=64)

    # model = ResUNetpp.build_resunetplusplus(num_classes, 16)

    # model = ConvNeXt.create_cityscapes_model()

    if torch.cuda.device_count() > 1:
        model = DataParallel(model)
    
    model = model.to(device)
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=learning_rate)
    
    best_val_loss = float('inf')
    best_iou = 0.0
    metrics_list = []  # 用于记录每个epoch的指标
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # 使用训练集训练
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
        
        # 使用测试集验证（注意：在实际应用中，应该使用独立的验证集而不是测试集）
        # 这里为了简化，直接使用测试集作为验证集
        val_loss, val_metrics = validate(model, test_loader, device, num_classes)
        
        epoch_time = time.time() - start_time
        
        # 记录当前epoch指标
        epoch_metrics = {
            'epoch': epoch+1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            **val_metrics
        }
        metrics_list.append(epoch_metrics)
        
        print(f"\nEpoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Time: {epoch_time:.1f}s")
        print(f"Validation Metrics: "
              f"Precision: {val_metrics['precision']:.4f}, "
              f"Recall: {val_metrics['recall']:.4f}, "
              f"Accuracy: {val_metrics['accuracy']:.4f}, "
              f"F1: {val_metrics['f1']:.4f}, "
              f"IoU: {val_metrics['iou']:.4f}, "
              f"Dice: {val_metrics['dice']:.4f}")
        
        # 保存最佳模型
        if val_metrics['iou'] > best_iou:  # 使用IoU作为选择最佳模型的标准
            best_iou = val_metrics['iou']
            model_state = model.module.state_dict() if isinstance(model, DataParallel) else model.state_dict()
            
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'metrics': val_metrics
            }, os.path.join(save_dir, f'{model.name}_best_model.pth'))
            print(f"保存最佳模型，验证IoU: {best_iou:.4f}")
    
    # 保存训练指标
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_excel(os.path.join(save_dir, f'{model.name}_Ablition_training_metrics.xlsx'), index=False)
    
    print(f"\n训练完成，最佳验证IoU: {best_iou:.4f}")
    
    # 绘制训练曲线
    plot_training_metrics(metrics_df, save_dir, model.name)

def plot_training_metrics(metrics_df, save_dir, model_name):
    """绘制训练指标曲线"""
    plt.figure(figsize=(15, 10))
    
    # 为每个指标创建子图
    metrics = ['train_loss', 'val_loss', 'iou', 'f1']
    titles = ['Training Loss', 'Validation Loss', 'IoU Score', 'F1 Score']
    
    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i+1)
        plt.plot(metrics_df['epoch'], metrics_df[metric], label=metric)
        plt.title(titles[i])
        plt.xlabel('Epoch')
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_name}_training_metrics.png'))
    plt.close()

# 在main调用处保持不变
if __name__ == "__main__":
    main()
