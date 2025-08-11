import os
import shutil
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import pandas as pd
import torch.optim as optim
import math
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, jaccard_score
from torchvision import datasets, transforms
from modules import *
from DataInstances import Cityscapes

def calculate_metrics(preds, targets, num_classes):

    if isinstance(preds, torch.Tensor):
        preds = preds.argmax(1).detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    preds_flat = preds.flatten()
    targets_flat = targets.flatten()

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
        'f1_score': f1,
        'iou': iou,
        'dice_coefficient': dice
    }

epochs = 50
batch_size = 8
learning_rate = 1e-3

os.makedirs("./Result", exist_ok=True)

root_dir = 'cityscapes'

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

small_train = Subset(train_dataset, indices=range(40))
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

small_train_loader = DataLoader(
    small_train,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)
    
test_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)
print("Data load successfully!")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ConvNeXt.create_cityscapes_model()

model = model.to(device)

class DiceFocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        
    def forward(self, pred, target):
        # Dice loss
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        dice_loss = 1 - dice
        
        # Focal loss
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        p_t = torch.exp(-bce)
        focal_loss = (1 - p_t) ** self.gamma * bce

        return self.alpha * dice_loss + (1 - self.alpha) * focal_loss.mean()

criterion = nn.CrossEntropyLoss(ignore_index=255)

optimizer = optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    weight_decay=0.01,
    betas=(0.9, 0.999),
    eps=1e-8
)

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

niter_per_ep = len(train_loader)
lr_schedule = cosine_scheduler(
    base_value=learning_rate,
    final_value=1e-6,
    epochs=epochs,
    niter_per_ep=niter_per_ep,
    warmup_epochs=5,
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
    torch.autograd.set_detect_anomaly(True)
    best_iou = 0.0
    iteration = 0

    metrics_df = pd.DataFrame(columns=['epoch', 'train_loss', 'val_loss', 'accuracy', 
                                      'precision', 'recall', 'f1_score', 'iou', 'dice_coefficient'])
    all_metrics = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        train_samples = 0
        progress_bar = tqdm(small_train_loader, desc=f'Epoch {epoch}/{epochs} [Train]')
        for batch_idx, (images, masks) in enumerate(progress_bar):

            it = iteration + batch_idx
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_schedule[it]
            
            images, masks = images.to(device), masks.to(device)
            masks = masks.long()

            
            outputs = model(images)
            if isinstance(outputs, list):
                main_output, aux_output = outputs
                loss = criterion(main_output, masks) + 0.4 * criterion(aux_output, masks)
            else:
                loss = criterion(outputs, masks)
            # print(f"Output = {outputs.size()}")
            # print(f"Masks = {masks.size()}")
            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            
            train_loss += loss.item() * images.size(0)
            train_samples += images.size(0)

            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{param_group['lr']:.6f}"
            })
        
        iteration += len(train_loader)
        avg_train_loss = train_loss / train_samples

        model.eval()
        val_loss = 0
        val_samples = 0
        all_preds = []
        all_masks = []
        
        with torch.no_grad():
            # val_progress = tqdm(test_loader, desc=f'Epoch {epoch}/{epochs} [Validation]')
            val_progress = tqdm(small_train_loader, desc=f'Epoch {epoch}/{epochs} [Validation]')
            for images, masks in val_progress:
                images, masks = images.to(device), masks.to(device)
                masks = masks.long()

                # 前向传播
                outputs = model(images)
                outputs = model(images)
                if isinstance(outputs, list):
                    main_output, aux_output = outputs
                    loss = criterion(main_output, masks) + 0.4 * criterion(aux_output, masks)
                else:
                    loss = criterion(outputs, masks)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)
                val_samples += images.size(0)

                preds = torch.argmax(outputs, dim=1)
 
                all_preds.append(preds.detach().cpu().numpy())
                all_masks.append(masks.detach().cpu().numpy())

                val_progress.set_postfix({
                    'val_loss': f"{loss.item():.4f}"
                })
        
        avg_val_loss = val_loss / val_samples

        all_preds = np.concatenate(all_preds, axis=0)
        all_masks = np.concatenate(all_masks, axis=0)

        if all_preds.ndim != 3:
            all_preds = all_preds.squeeze(1)
        if all_masks.ndim != 3:
            all_masks = all_masks.squeeze(1)
        
        metrics = calculate_metrics(all_masks, all_preds, 19)

        metrics_history['train_loss'].append(avg_train_loss)
        metrics_history['val_loss'].append(avg_val_loss)
        for k, v in metrics.items():
            metrics_history[k].append(v)

        print(f"\nEpoch {epoch}/{epochs}:")
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        print(f"Accuracy: {metrics['accuracy']:.4f}, Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}, F1: {metrics['f1_score']:.4f}")
        print(f"IoU: {metrics['iou']:.4f}, Dice: {metrics['dice_coefficient']:.4f}")

        epoch_metrics = {
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            **metrics
        }
        all_metrics.append(epoch_metrics)
        metrics_df = pd.DataFrame(all_metrics)
        metrics_df.to_csv(f"./Result/metrics_{model.name}_history.csv", index=False)

        if metrics['iou'] > best_iou:
            best_iou = metrics['iou']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': best_iou,
                'val_loss': avg_val_loss,
            }, f"./Result/best_model_iou_{best_iou:.4f}.pth")
            print(f"Saved new best model with IoU: {best_iou:.4f}")
        
    
    print("Training completed!")