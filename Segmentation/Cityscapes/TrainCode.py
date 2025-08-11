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
from DataInstances import *
from tqdm import tqdm
from torch.nn.parallel import DataParallel


def get_data_loaders(root_dir, batch_size=2, num_workers=4):

    train_dataset = Cityscapes.Cityscapes(
        root_dir=root_dir,
        split='train',
        transform=Cityscapes.Cityscapes.get_default_transform()
    )
    
    val_dataset = Cityscapes.Cityscapes(
        root_dir=root_dir,
        split='val',
        transform=Cityscapes.Cityscapes.get_default_transform()
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=lambda batch: tuple(zip(*batch))
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda batch: tuple(zip(*batch))
    )
    
    return train_loader, val_loader

def compute_loss(outputs, targets):
    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs, targets.long())
    
    return loss

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10):
    model.train()
    total_loss = 0
    num_batches = len(data_loader)
    
    pbar = tqdm(enumerate(data_loader), total=num_batches, desc=f"Epoch {epoch+1}")
    
    for i, (images, targets) in pbar:
        images = [image.to(device) for image in images]
        targets = [target.to(device) for target in targets]
        
        outputs = model(torch.stack(images))
        print(f"Training outputs = {outputs.size()}")
        
        loss = compute_loss(outputs, torch.stack(targets))
        total_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        pbar.set_postfix(loss=loss.item())
    
    return total_loss / num_batches

def validate(model, data_loader, device):
    model.eval()
    total_loss = 0
    num_batches = len(data_loader)
    
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Validating"):
            images = [image.to(device) for image in images]

            targets = [target.to(device) for target in targets]
            
            outputs = model(torch.stack(images))
            
            loss = compute_loss(outputs, torch.stack(targets))
            total_loss += loss.item()
    
    return total_loss / num_batches

def main():
    root_dir = 'cityscapes'
    num_classes = 19
    num_epochs = 100
    batch_size = 8
    learning_rate = 0.0001
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = 'checkpoints'
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"可用GPU数量: {torch.cuda.device_count()}")
    print(f"使用设备: {device}")
    print(f"训练配置: {num_epochs} epochs, 批次大小 {batch_size}, 学习率 {learning_rate}")
    
    model = RMT.RMT_Segmentation(
        num_classes=num_classes, in_chans=3, out_indices=(0, 1, 2, 3),
        embed_dims=[112, 224, 448, 640], depths=[4, 8, 25, 8], num_heads=[7, 7, 14, 20],
        init_values=[2, 2, 2, 2], heads_ranges=[6, 6, 6, 6], mlp_ratios=[4, 4, 3, 3], drop_path_rate=0.5
    )
    
    if torch.cuda.device_count() > 1:
        print(f"使用 {torch.cuda.device_count()} 个GPU进行训练")
        model = DataParallel(model)
    
    model = model.to(device)
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=learning_rate)
    
    train_loader, val_loader = get_data_loaders(root_dir, batch_size)
    
    print(f"训练集大小: {len(train_loader.dataset)}")
    print(f"验证集大小: {len(val_loader.dataset)}")
    
    best_val_loss = float('inf')
    train_history = []
    val_history = []
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
        train_history.append(train_loss)
        
        val_loss = validate(model, val_loader, device)
        val_history.append(val_loss)
        
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Time: {epoch_time:.1f}s")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_state = model.module.state_dict() if isinstance(model, DataParallel) else model.state_dict()
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_history': train_history,
                'val_history': val_history
            }
            torch.save(checkpoint, os.path.join(save_dir, 'best_model.pth'))
            print(f"保存最佳模型，验证损失: {val_loss:.4f}")
        
        if (epoch + 1) % 5 == 0:
            torch.save(checkpoint, os.path.join(save_dir, f'model_epoch_{epoch+1}.pth'))
    
    print("训练完成!")
    
    model_state = model.module.state_dict() if isinstance(model, DataParallel) else model.state_dict()
    final_checkpoint = {
        'epoch': num_epochs,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_history': train_history,
        'val_history': val_history
    }
    torch.save(final_checkpoint, os.path.join(save_dir, 'final_model.pth'))
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(train_history, label='Train Loss')
    plt.plot(val_history, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'loss_curve.png'))

if __name__ == "__main__":
    main()