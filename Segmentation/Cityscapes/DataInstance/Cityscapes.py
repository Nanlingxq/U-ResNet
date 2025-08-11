import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class Cityscapes(Dataset):
    def __init__(self, root_dir, split='train', transform=None):

        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        # 确定使用的文本文件
        if split == 'train':
            image_list_file = os.path.join(root_dir, 'trainImages.txt')
            instance_list_file = os.path.join(root_dir, 'trainInstances.txt')
        elif split == 'val':
            image_list_file = os.path.join(root_dir, 'valImages.txt')
            instance_list_file = os.path.join(root_dir, 'valInstances.txt')
        else:
            image_list_file = os.path.join(root_dir, 'testImages.txt')
            instance_list_file = None

        # 读取图像路径列表
        with open(image_list_file, 'r') as f:
            self.image_paths = [os.path.join(root_dir, line.strip()) for line in f.readlines() if line.strip()]

        # 读取实例标签路径列表
        self.instance_paths = []
        if instance_list_file and os.path.exists(instance_list_file):
            with open(instance_list_file, 'r') as f:
                self.instance_paths = [os.path.join(root_dir, line.strip()) for line in f.readlines() if line.strip()]

        # 确保图像和标签数量匹配
        if self.instance_paths:
            assert len(self.image_paths) == len(self.instance_paths), \
                f"图像数量({len(self.image_paths)})和标签数量({len(self.instance_paths)})不匹配"

        # 打印信息用于调试
        print(f"加载{split}数据集: {len(self.image_paths)}个样本")
        if self.instance_paths:
            print(f"其中{len(self.instance_paths)}个有实例标签")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 加载原始图像
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        # 初始化实例标签
        instance_label = None
        
        # 如果有对应的实例标签，加载它
        if idx < len(self.instance_paths) and self.instance_paths[idx]:
            instance_path = self.instance_paths[idx]
            if os.path.exists(instance_path):
                instance_label = Image.open(instance_path)
                instance_label = np.array(instance_label)
                
                # 只保留车辆类别的实例
                vehicle_classes = [12, 13, 14, 15, 17, 18]
                category_map = instance_label // 1000
                vehicle_mask = np.isin(category_map, vehicle_classes)
                new_instance_label = np.zeros_like(instance_label, dtype=np.uint8)
                new_instance_label[vehicle_mask] = 1
                instance_label = new_instance_label
        
        # 应用变换
        if self.transform:
            if instance_label is not None:
                transformed = self.transform(image=image, instance=instance_label)
                image = transformed['image']
                instance_label = transformed['instance']
            else:
                # 没有标签时只变换图像
                transformed = self.transform(image=image)
                image = transformed['image']
        
        if not self.transform:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            if instance_label is not None:
                instance_label = torch.from_numpy(instance_label).long()
        
        if instance_label is not None:
            return image, instance_label
        else:
            return image, torch.zeros(1, dtype=torch.long)

    @staticmethod
    def get_default_transform(image_size=(512, 1024)):

        base_transform = A.Compose([
            A.Resize(height=image_size[0], width=image_size[1]),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ], additional_targets={'instance': 'mask'})

        return base_transform

    def analyze_labels(self):
        # 收集所有标签值
        all_values = []
        for i in range(len(self)):
            _, label = self[i]
            if label.dim() > 0:  # 确保不是空标签
                all_values.extend(torch.unique(label.long()).tolist())
        
        # 转换为numpy数组以便处理
        all_values = np.array(all_values)
        all_values = all_values[all_values != 0]
        
        # 计算类别ID
        category_ids = all_values // 1000
        
        unique_categories = np.unique(category_ids)
        
        print(f"发现 {len(unique_categories)} 个唯一类别")
        print(f"类别ID: {unique_categories}")
        
        return unique_categories
        
class CityscapesSemantic(Dataset):  # 重命名类以反映用途
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        

        # 确定使用的文本文件
        if split == 'train':
            image_list_file = os.path.join(root_dir, 'trainImages.txt')
            label_list_file = os.path.join(root_dir, 'trainLabels.txt')  # 语义标签文件
        elif split == 'val':
            image_list_file = os.path.join(root_dir, 'valImages.txt')
            label_list_file = os.path.join(root_dir, 'valLabels.txt')    # 语义标签文件
        elif split == 'trainval':  # 新增trainval模式
            # 加载训练集
            train_image_file = os.path.join(root_dir, 'trainImages.txt')
            train_label_file = os.path.join(root_dir, 'trainLabels.txt')
            
            # 加载验证集
            val_image_file = os.path.join(root_dir, 'valImages.txt')
            val_label_file = os.path.join(root_dir, 'valLabels.txt')
            
            # 读取训练集路径
            with open(train_image_file, 'r') as f:
                train_images = [os.path.join(root_dir, line.strip()) for line in f.readlines() if line.strip()]
            with open(train_label_file, 'r') as f:
                train_labels = [os.path.join(root_dir, line.strip()) for line in f.readlines() if line.strip()]
                
            # 读取验证集路径
            with open(val_image_file, 'r') as f:
                val_images = [os.path.join(root_dir, line.strip()) for line in f.readlines() if line.strip()]
            with open(val_label_file, 'r') as f:
                val_labels = [os.path.join(root_dir, line.strip()) for line in f.readlines() if line.strip()]
            
            # 合并路径列表
            self.image_paths = train_images + val_images
            self.label_paths = train_labels + val_labels
        else:
            image_list_file = os.path.join(root_dir, 'testImages.txt')
            label_list_file = None

        # 读取图像路径列表
        if split != 'trainval':
            with open(image_list_file, 'r') as f:
                self.image_paths = [os.path.join(root_dir, line.strip()) for line in f.readlines() if line.strip()]

        # 读取语义标签路径列表
        if split != 'trainval':
            self.label_paths = []
            if label_list_file and os.path.exists(label_list_file):
                with open(label_list_file, 'r') as f:
                    self.label_paths = [os.path.join(root_dir, line.strip()) for line in f.readlines() if line.strip()]

        # 确保图像和标签数量匹配
        if self.label_paths:
            assert len(self.image_paths) == len(self.label_paths), \
                f"图像数量({len(self.image_paths)})和标签数量({len(self.label_paths)})不匹配"

        # 可选：定义Cityscapes的19个可训练类别
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 
                              23, 24, 25, 26, 27, 28, 31, 32, 33]
        self.class_map = dict(zip(self.valid_classes, range(19)))
        self.ignore_index = 255

        print(f"加载{split}数据集: {len(self.image_paths)}个样本")
        if self.label_paths:
            print(f"其中{len(self.label_paths)}个有语义标签")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 加载原始图像
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        # 初始化语义标签
        semantic_label = None
        
        # 如果有对应的语义标签，加载它
        if idx < len(self.label_paths) and self.label_paths[idx]:
            label_path = self.label_paths[idx]
            if os.path.exists(label_path):
                semantic_label = Image.open(label_path)
                semantic_label = np.array(semantic_label)
                
                # 可选：映射到可训练类别
                mask = np.zeros_like(semantic_label, dtype=np.uint8) + self.ignore_index
                for valid_class in self.valid_classes:
                    mask[semantic_label == valid_class] = self.class_map[valid_class]
                semantic_label = mask
        
        # 应用变换
        if self.transform:
            if semantic_label is not None:
                transformed = self.transform(image=image, mask=semantic_label)
                image = transformed['image']
                semantic_label = transformed['mask']  # 注意键名改为'mask'
                if semantic_label.dim() == 3 and semantic_label.size() == 1:
                    semantic_label = semantic_label.squeeze()
                semantic_label = semantic_label.long()
            else:
                transformed = self.transform(image=image)
                image = transformed['image']
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            if semantic_label is not None:
                if semantic_label.ndim == 3 and semantic_label.shape[2] == 1:
                    semantic_label = semantic_label.squeeze(2)
                semantic_label = torch.from_numpy(semantic_label).long()
        
        if semantic_label is not None:
            return image, semantic_label
        else:
            return image, torch.zeros(1, dtype=torch.long)  # 无标签时返回空张量

    @staticmethod
    def get_default_transform(image_size=(512, 1024)):
        # 语义分割专用变换
        base_transform = A.Compose([
            A.Resize(height=image_size[0], width=image_size[1]),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ], additional_targets={'mask': 'mask'})  # 使用'mask'作为标签键名

        return base_transform

    def analyze_labels(self):
        # 收集所有标签值
        all_values = []
        for i in range(min(100, len(self))):  # 限制样本数量
            _, label = self[i]
            if label.dim() > 0:  # 确保不是空标签
                unique_vals = torch.unique(label.long())
                all_values.extend(unique_vals[unique_vals != self.ignore_index].tolist())
        
        # 计算统计信息
        unique_vals, counts = np.unique(all_values, return_counts=True)
        
        print(f"发现 {len(unique_vals)} 个唯一类别")
        for val, count in zip(unique_vals, counts):
            print(f"类别 {val}: {count} 像素 ({count/len(all_values)*100:.2f}%)")
        
        return unique_vals

def save_mask_overlay(image, label, output_path, opacity=0.5):
    
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    denorm_image = image * std + mean
    denorm_image = denorm_image.permute(1, 2, 0).numpy() * 255.0
    denorm_image = denorm_image.astype(np.uint8)
    
    # 转换为PIL图像
    original_img = Image.fromarray(denorm_image)
    
    # 创建彩色掩码
    label_np = label.numpy()
    unique_ids = np.unique(label_np)
    
    # 创建RGB掩码图像
    colored_mask = np.zeros((*label_np.shape, 3), dtype=np.uint8)
    
    # 为每个唯一ID分配随机颜色（跳过背景0）
    color_map = {}
    for id_val in unique_ids:
        if id_val == 0:  # 背景保持透明（黑色）
            color_map[id_val] = [0, 0, 0, 0]  # RGBA，A=0表示完全透明
        else:
            # 生成鲜艳的颜色
            color_map[id_val] = list(np.random.randint(50, 255, size=3)) + [int(255 * opacity)]  # RGBA
    
    # 应用颜色映射
    for i in range(label_np.shape[0]):
        for j in range(label_np.shape[1]):
            if label_np[i, j] in color_map:
                colored_mask[i, j] = color_map[label_np[i, j]][:3]  # 只取RGB部分
    
    # 创建掩码图像
    mask_img = Image.fromarray(colored_mask).convert("RGBA")
    
    # 创建叠加图像
    overlay = original_img.copy().convert("RGBA")
    overlay.paste(mask_img, (0, 0), mask_img)  # 使用alpha通道叠加
    
    # 保存结果
    overlay.save(output_path)
    print(f"已保存叠加图像到: {output_path}")

# 主程序
if __name__ == "__main__":
    root_dir = 'cityscapes'  # 替换为您的实际路径
    
    # 创建数据集实例（使用默认变换）
    dataset = Cityscapes(
        root_dir=root_dir,
        split='train',
        transform=Cityscapes.get_default_transform()
    )

    # dataset.analyze_labels()
    
    # 确保输出目录存在
    output_dir = "mask_overlays"
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理并保存多个样本
    num_samples = 5  # 要处理的样本数量
    
    for i in range(min(num_samples, len(dataset))):
        # 获取样本
        image, instance_label = dataset[i]
        
        # 打印样本信息
        print(f"\n样本 {i}:")
        print(f"图像路径: {dataset.image_paths[i]}")
        if i < len(dataset.instance_paths):
            print(f"标签路径: {dataset.instance_paths[i]}")
        print(f"图像形状: {image.shape}")
        print(f"标签形状: {instance_label.shape}")

        # 创建叠加图像并保存
        output_path = os.path.join(output_dir, f"mask_overlay_{i}.png")
        save_mask_overlay(image, instance_label, output_path, opacity=0.6)
        
        # 单独保存原始图像
        original_path = os.path.join(output_dir, f"original_{i}.png")
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        denorm_image = image * std + mean
        denorm_image = denorm_image.permute(1, 2, 0).numpy() * 255.0
        denorm_image = denorm_image.astype(np.uint8)
        Image.fromarray(denorm_image).save(original_path)
        
        # 单独保存掩码图像
        mask_path = os.path.join(output_dir, f"mask_{i}.png")
        label_np = instance_label.numpy()
        # 归一化到0-255范围以便可视化
        normalized_mask = (label_np / label_np.max() * 255).astype(np.uint8) if label_np.max() > 0 else label_np.astype(np.uint8)
        Image.fromarray(normalized_mask).save(mask_path)

    print(f"\n处理完成! 结果保存在 '{output_dir}' 目录中")