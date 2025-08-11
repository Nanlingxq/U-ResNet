import os
import shutil
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import pandas as pd
import torch.optim as optim
import math
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from tqdm import tqdm
from IceSeg import IceSegDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, jaccard_score
from torchvision import datasets, transforms
from modules import *

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
# 修复损失函数实现
class FixedWeightedDiceFocalLoss(nn.Module):
    def __init__(self, alpha=0.7, gamma=2.0, smooth=1e-6, pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.pos_weight = pos_weight
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')

    def forward(self, pred, target):
        # 确保尺寸匹配
        if pred.size() != target.size():
            pred = F.interpolate(pred, size=target.shape[-2:], mode='bilinear', align_corners=False)
        
        # 避免原地操作
        pred_sigmoid = torch.sigmoid(pred)
        intersection = torch.sum(pred_sigmoid * target)
        
        # 避免除零错误
        denominator = torch.sum(pred_sigmoid) + torch.sum(target)
        dice = (2. * intersection + self.smooth) / (denominator + self.smooth)
        dice_loss = 1 - dice
        
        # Focal loss部分
        bce = self.bce_loss(pred, target)
        p_t = torch.exp(-bce)
        focal_loss = (1 - p_t) ** self.gamma * bce
        
        # 返回平均损失
        return self.alpha * dice_loss + (1 - self.alpha) * focal_loss.mean()

def calculate_class_weights(loader, device):
    total_pixels = 0
    positive_pixels = 0
    
    for _, masks in loader:
        positive_pixels += masks.sum().item()
        total_pixels += masks.numel()
    
    positive_ratio = positive_pixels / total_pixels
    negative_ratio = 1 - positive_ratio
    pos_weight = torch.tensor([negative_ratio / (positive_ratio + 1e-6)]).to(device)
    
    print(f"Class balance: Positive={positive_ratio:.4f}, Negative={negative_ratio:.4f}")
    print(f"Using pos_weight: {pos_weight.item():.4f}")
    return pos_weight

def find_optimal_threshold(model, loader, device):
    """通过验证集寻找最佳阈值"""
    model.eval()
    thresholds = np.linspace(0.1, 0.9, 17)  # 17个点从0.1到0.9
    best_threshold = 0.5
    best_f1 = 0
    
    with torch.no_grad():
        all_preds = []
        all_targets = []
        
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            
            # 修改这里：提取主输出
            if isinstance(outputs, list) or isinstance(outputs, tuple):
                # 模型返回多个输出，取第一个（主输出）
                main_output = outputs[0]
            else:
                main_output = outputs
                
            pred_probs = torch.sigmoid(main_output).cpu().numpy().flatten()
            all_preds.append(pred_probs)
            all_targets.append(masks.cpu().numpy().flatten())
        
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        
        for th in thresholds:
            preds = (all_preds > th).astype(float)
            _, _, f1, _ = precision_recall_fscore_support(
                all_targets, preds, average='binary', zero_division=0
            )
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = th
    
    print(f"Optimal threshold found: {best_threshold:.4f} (F1={best_f1:.4f})")
    return best_threshold
# 定义余弦退火学习率调度器
def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=5, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)
    
    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
    
    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule

# 超参数设置
epochs = 50
batch_size = 8
learning_rate = 5e-4

# 创建结果目录
os.makedirs("./Result", exist_ok=True)

image_transform = transforms.Compose([
    transforms.ToPILImage(), 
    transforms.RandomCrop(640, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

mask_transform = transforms.Compose([
    transforms.ToTensor(),
])

# 加载数据集
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

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)
print("Data load successfully!")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 使用预训练模型
model = ConvNeXt.ConvNeXtUPerNet(
    num_classes=1,  # 分割类别数
    backbone='convnext_base',  # 使用base版本
    in_chans=3,  # 输入通道数
    img_size=640,  # 输入图像尺寸
    with_aux_head=True
)
model = model.to(device)

# 计算类别权重
pos_weight = calculate_class_weights(train_loader, device)
criterion = FixedWeightedDiceFocalLoss(
    alpha=0.7, 
    gamma=3,
    pos_weight=pos_weight
)

# 优化器配置 - 使用更保守的设置
optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-4,  # 降低初始学习率
    weight_decay=0.01,  # 降低权重衰减
    betas=(0.9, 0.999),
    eps=1e-8
)

# 创建学习率调度器
niter_per_ep = len(train_loader)
lr_schedule = cosine_scheduler(
    base_value=1e-4,  # 降低基础学习率
    final_value=1e-6,
    epochs=epochs,
    niter_per_ep=niter_per_ep,
    warmup_epochs=3,  # 缩短warmup
    start_warmup_value=1e-6
)

metrics_history = {
    'train_loss': [],
    'val_loss': [],
    'accuracy': [],
    'precision': [],
    'recall': [],
    'f1_score': [],
    'iou': [],
    'dice_coefficient': []
}

if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)  # 启用异常检测
    best_iou = 0.0
    iteration = 0  # 全局迭代计数器
    
    # 梯度缩放用于混合精度训练
    scaler = torch.amp.GradScaler('cuda')
    
    # 创建指标记录文件
    metrics_df = pd.DataFrame(columns=['epoch', 'train_loss', 'val_loss', 'accuracy', 
                                      'precision', 'recall', 'f1_score', 'iou', 'dice_coefficient'])
    all_metrics = []
    
    # 初始阈值设定
    optimal_threshold = 0.5
    # 在训练开始前寻找初始阈值
    optimal_threshold = find_optimal_threshold(model, test_loader, device)
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        train_samples = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}/{epochs} [Train]')
        
        total_grad_norm = 0
        grad_count = 0
        
        for batch_idx, (images, masks) in enumerate(progress_bar):
            # 更新学习率
            it = iteration + batch_idx
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_schedule[it]
            
            images, masks = images.to(device), masks.to(device)
            
            # 前向传播
            outputs = model(images)
            
            # 处理多个输出
            if isinstance(outputs, list) or isinstance(outputs, tuple):
                main_output = outputs[0]
                if len(outputs) > 1:
                    aux_output = outputs[1]
                    
                    # 确保尺寸匹配
                    if main_output.size() != masks.size():
                        main_output = F.interpolate(main_output, size=masks.shape[-2:], mode='bilinear', align_corners=False)
                    if aux_output.size() != masks.size():
                        aux_output = F.interpolate(aux_output, size=masks.shape[-2:], mode='bilinear', align_corners=False)
                    
                    # 计算损失
                    main_loss = criterion(main_output, masks)
                    aux_loss = criterion(aux_output, masks)
                    loss = main_loss + 0.4 * aux_loss
                else:
                    if main_output.size() != masks.size():
                        main_output = F.interpolate(main_output, size=masks.shape[-2:], mode='bilinear', align_corners=False)
                    loss = criterion(main_output, masks)
            else:
                if outputs.size() != masks.size():
                    outputs = F.interpolate(outputs, size=masks.shape[-2:], mode='bilinear', align_corners=False)
                loss = criterion(outputs, masks)
            
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            if len(grads) == 0:
                print("Warning: No gradients found, skipping update")
                continue
                
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            total_grad_norm += grad_norm.item()
            grad_count += 1
            
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            train_samples += images.size(0)
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{param_group['lr']:.6f}",
            })
        
        # 打印梯度信息
        avg_grad_norm = total_grad_norm / grad_count
        print(f"Epoch {epoch} - Average gradient norm: {avg_grad_norm:.4f}")
        
        iteration += len(train_loader)  # 更新全局迭代计数器
        avg_train_loss = train_loss / train_samples
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_samples = 0
        all_preds = []
        all_masks = []
        
        with torch.no_grad():
            val_progress = tqdm(test_loader, desc=f'Epoch {epoch}/{epochs} [Validation]')
            for images, masks in val_progress:
                images, masks = images.to(device), masks.to(device)
                
                # 前向传播
                outputs = model(images)
                
                # 修改这里：提取主输出
                if isinstance(outputs, list) or isinstance(outputs, tuple):
                    main_output = outputs[0]
                else:
                    main_output = outputs
                    
                # 使用主输出计算损失
                loss = criterion(main_output, masks)
                val_loss += loss.item() * images.size(0)
                val_samples += images.size(0)
                
                # 使用动态调整的阈值
                preds = torch.sigmoid(main_output)
                preds = (preds > optimal_threshold).float()
        
                all_preds.append(preds.detach().cpu().numpy())
                all_masks.append(masks.detach().cpu().numpy())
                
                # 更新进度条
                val_progress.set_postfix({
                    'val_loss': f"{loss.item():.4f}"
                })
        
        # 每5个epoch重新计算一次阈值
        if epoch % 5 == 0 or epoch == 1:
            optimal_threshold = find_optimal_threshold(model, test_loader, device)
        
        avg_val_loss = val_loss / val_samples
        
        # 计算指标
        all_preds = np.concatenate(all_preds, axis=0)
        all_masks = np.concatenate(all_masks, axis=0)

        # 调整维度: (batch, 1, H, W) -> (batch, H, W)
        all_preds = all_preds.squeeze(1)
        all_masks = all_masks.squeeze(1)

        # 如果形状是 (batch, 1, H, W)，需要进一步调整
        if all_preds.ndim == 4 and all_preds.shape[1] == 1:
            all_preds = all_preds.squeeze(1)
        if all_masks.ndim == 4 and all_masks.shape[1] == 1:
            all_masks = all_masks.squeeze(1)

        metrics = compute_segmentation_metrics(all_masks, all_preds)
        
        # 记录指标
        metrics_history['train_loss'].append(avg_train_loss)
        metrics_history['val_loss'].append(avg_val_loss)
        for k, v in metrics.items():
            metrics_history[k].append(v)
        
        # 打印指标
        print(f"\nEpoch {epoch}/{epochs}:")
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        print(f"Accuracy: {metrics['accuracy']:.4f}, Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}, F1: {metrics['f1_score']:.4f}")
        print(f"IoU: {metrics['iou']:.4f}, Dice: {metrics['dice_coefficient']:.4f}")
        print(f"Used threshold: {optimal_threshold:.4f}")
        
        # 保存指标到DataFrame
        epoch_metrics = {
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'threshold': optimal_threshold,
            **metrics
        }
        all_metrics.append(epoch_metrics)
        metrics_df = pd.DataFrame(all_metrics)
        metrics_df.to_csv(f"./Result/{model.name}_metrics_history.csv", index=False)
        
        # 保存最佳模型
        if metrics['iou'] > best_iou:
            best_iou = metrics['iou']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': best_iou,
                'val_loss': avg_val_loss,
                'threshold': optimal_threshold
            }, f"./Result/best_model_iou_{best_iou:.4f}.pth")
            print(f"Saved new best model with IoU: {best_iou:.4f}")
        
        # 定期保存检查点
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': best_iou,
                'val_loss': avg_val_loss,
            }, f"./Result/checkpoint_epoch_{epoch}.pth")
    
    print("Training completed!")