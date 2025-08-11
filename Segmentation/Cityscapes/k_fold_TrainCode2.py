import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
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

def get_data_loaders(root_dir, batch_size=2, num_workers=4):
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
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    loss = criterion(outputs, targets.long())
    return loss

def calculate_metrics(preds, targets, num_classes):
    preds_flat = preds.argmax(1).flatten().cpu().numpy()
    targets_flat = targets.flatten().cpu().numpy()

    valid_mask = targets_flat != 255
    preds_flat = preds_flat[valid_mask]
    targets_flat = targets_flat[valid_mask]

    precision = precision_score(targets_flat, preds_flat, average='macro', zero_division=0)
    recall = recall_score(targets_flat, preds_flat, average='macro', zero_division=0)
    accuracy = accuracy_score(targets_flat, preds_flat)
    f1 = f1_score(targets_flat, preds_flat, average='macro', zero_division=0)

    iou = jaccard_score(targets_flat, preds_flat, average='macro', zero_division=0)

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
    model.train()
    total_loss = 0
    num_batches = len(data_loader)
    
    pbar = tqdm(data_loader, total=num_batches, desc=f"Epoch {epoch+1}")
    
    for images, targets in pbar:
        images = images.to(device)
        targets = targets.to(device)

        outputs = model(images)
        
        loss = compute_loss(outputs, targets)
        total_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        pbar.set_postfix(loss=loss.item())
    
    return total_loss / num_batches

def validate(model, data_loader, device, num_classes):
    model.eval()
    total_loss = 0
    num_batches = len(data_loader)

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

            batch_metrics = calculate_metrics(outputs, targets, num_classes)

            for key in metrics:
                metrics[key] += batch_metrics[key]

    for key in metrics:
        metrics[key] /= num_batches
    
    return total_loss / num_batches, metrics

def main():
    root_dir = 'cityscapes'
    num_classes = 19
    num_epochs = 30
    batch_size = 16
    learning_rate = 0.0001
    n_splits = 10
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = 'checkpoints'
    os.makedirs(save_dir, exist_ok=True)

    full_dataset = Cityscapes.CityscapesSemantic(
        root_dir=root_dir,
        split='trainval',
        transform=Cityscapes.CityscapesSemantic.get_default_transform()
    )

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []
    all_folds_metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(full_dataset)):
        print(f"\n{'='*40}")
        print(f"开始训练 Fold {fold+1}/{n_splits}")
        print(f"{'='*40}")

        train_subset = torch.utils.data.Subset(full_dataset, train_idx)
        val_subset = torch.utils.data.Subset(full_dataset, val_idx)

        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # 初始化模型（每折使用独立模型）
        # model = RMT.RMT_Segmentation(
        #     num_classes=num_classes, in_chans=3, out_indices=(0, 1, 2, 3),
        #     embed_dims=[56, 112, 224, 320], depths=[4, 8, 25, 8], num_heads=[7, 7, 14, 20],
        #     init_values=[2, 2, 2, 2], heads_ranges=[6, 6, 6, 6], mlp_ratios=[4, 4, 3, 3], drop_path_rate=0.5
        # )
        
        # model = U_ResNet.UResNetNewEUCB(seg_classes=num_classes, in_channels=3, base_c=64)

        # model = U_ResNet.UNet(seg_classes=num_classes, in_channels=3, base_c=64)

        # model = ResUNetpp.build_resunetplusplus(num_classes, 16)

        model = ConvXet.ConvNeXtDeepLab(num_classes=19)

        if torch.cuda.device_count() > 1:
            model = DataParallel(model)
        
        model = model.to(device)
        
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.Adam(params, lr=learning_rate)
        
        best_val_loss = float('inf')
        fold_metrics = []
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
            val_loss, val_metrics = validate(model, val_loader, device, num_classes)
            
            epoch_time = time.time() - start_time

            epoch_metrics = {
                'fold': fold+1,
                'epoch': epoch+1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                **val_metrics
            }
            fold_metrics.append(epoch_metrics)
            
            print(f"\nFold {fold+1} Epoch {epoch+1}/{num_epochs} - "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                  f"Time: {epoch_time:.1f}s")
            print(f"Validation Metrics: "
                  f"Precision: {val_metrics['precision']:.4f}, "
                  f"Recall: {val_metrics['recall']:.4f}, "
                  f"Accuracy: {val_metrics['accuracy']:.4f}, "
                  f"F1: {val_metrics['f1']:.4f}, "
                  f"IoU: {val_metrics['iou']:.4f}, "
                  f"Dice: {val_metrics['dice']:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model_state = model.module.state_dict() if isinstance(model, DataParallel) else model.state_dict()
                
                torch.save({
                    'fold': fold+1,
                    'epoch': epoch+1,
                    'model_state_dict': model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'metrics': val_metrics
                }, os.path.join(save_dir, f'{model.name}_best_model_fold_{fold+1}.pth'))
                print(f"保存Fold {fold+1}最佳模型，验证损失: {val_loss:.4f}")

        fold_df = pd.DataFrame(fold_metrics)
        fold_df.to_excel(os.path.join(save_dir, f'{model.name}_fold_{fold+1}_metrics.xlsx'), index=False)
        all_folds_metrics.append(fold_metrics)

        best_epoch_idx = fold_df['val_loss'].idxmin()
        fold_results.append({
            'fold': fold+1,
            'best_epoch': int(fold_df.loc[best_epoch_idx, 'epoch']),
            'best_val_loss': fold_df.loc[best_epoch_idx, 'val_loss'],
            'best_iou': fold_df.loc[best_epoch_idx, 'iou'],
            'best_f1': fold_df.loc[best_epoch_idx, 'f1']
        })

    results_df = pd.DataFrame(fold_results)
    results_df.to_excel(os.path.join(save_dir, 'kfold_results_summary.xlsx'), index=False)

    avg_val_loss = np.mean([r['best_val_loss'] for r in fold_results])
    avg_iou = np.mean([r['best_iou'] for r in fold_results])
    avg_f1 = np.mean([r['best_f1'] for r in fold_results])
    
    print(f"\n{'='*50}")
    print(f"K折交叉验证完成 (K={n_splits})")
    print(f"平均最佳验证损失: {avg_val_loss:.4f}")
    print(f"平均最佳IoU: {avg_iou:.4f}")
    print(f"平均最佳F1: {avg_f1:.4f}")
    print(f"{'='*50}")

    final_model_state = model.module.state_dict() if isinstance(model, DataParallel) else model.state_dict()
    torch.save(final_model_state, os.path.join(save_dir, 'final_model.pth'))

    plot_kfold_metrics(all_folds_metrics, save_dir)

def plot_kfold_metrics(all_folds_metrics, save_dir):
    plt.figure(figsize=(15, 10))

    metrics = ['train_loss', 'val_loss', 'iou', 'f1']
    titles = ['Training Loss', 'Validation Loss', 'IoU Score', 'F1 Score']
    
    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i+1)
        for fold_idx, fold_metrics in enumerate(all_folds_metrics):
            epochs = [m['epoch'] for m in fold_metrics]
            values = [m[metric] for m in fold_metrics]
            plt.plot(epochs, values, label=f'Fold {fold_idx+1}')
        
        plt.title(titles[i])
        plt.xlabel('Epoch')
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'kfold_training_metrics.png'))
    plt.close()

if __name__ == "__main__":
    main()
