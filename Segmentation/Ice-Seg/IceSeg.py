import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import yaml
from torchvision import transforms as transforms
import matplotlib.pyplot as plt

class IceSegDataset(Dataset):
    def __init__(self, yaml_path, dataset_type, image_transforms=None, mask_transforms=None):
        """
        初始化数据集。
        :param yaml_path: YAML 配置文件路径
        :param dataset_type: 数据集类型，可以是 'train', 'valid', 或 'test'
        :param transforms: 图像和掩码的转换
        """
        with open(yaml_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # 根据 YAML 文件内容获取数据集路径
        base_dir = os.path.dirname(yaml_path)
        self.images_dir = os.path.join(base_dir, dataset_type, 'images')
        self.masks_dir = os.path.join(base_dir, dataset_type, 'labels')
        self.image_transforms = image_transforms
        self.mask_transforms = mask_transforms

        # 获取图像文件路径列表
        self.image_files = [f for f in os.listdir(self.images_dir) if f.endswith(('.jpg', '.png'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 加载图像
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 加载对应的掩码
        mask_name = img_name[0:-4] + '.txt'
        mask_path = os.path.join(self.masks_dir, mask_name)
        mask = self._parse_mask(mask_path, img.shape[0], img.shape[1], img_name, mask_name)

        # 应用图像数据增强
        if self.image_transforms:
            img = self.image_transforms(img)
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        # 应用掩码数据增强
        if self.mask_transforms:
            mask = self.mask_transforms(mask)
        else:
            mask = torch.from_numpy(mask).unsqueeze(0).float() / 255.0

        return img, mask

    def _parse_mask(self, txt_path, img_height, img_width, img_name, mask_name):
        mask = np.zeros((img_height, img_width), dtype=np.uint8)

        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                lines = f.readlines()

            for line in lines:
                coords = list(map(float, line.strip().split()))
                if len(coords) < 4:  # 至少需要类别+至少一个坐标点（2个值）
                    # print(f"Skipping invalid line in {mask_name}: {line.strip()}")
                    continue

                class_id = coords[0]  # 提取类别标签
                polygon_coords = coords[1:]  # 跳过类别标签，只处理坐标

                points = []
                for i in range(0, len(polygon_coords), 2):
                    if i + 1 >= len(polygon_coords):
                        # print(f"Skipping incomplete point in {mask_name}: {line.strip()}")
                        break
                    x = int(polygon_coords[i] * img_width)
                    y = int(polygon_coords[i + 1] * img_height)
                    points.append((x, y))
                else:
                    if len(points) >= 3:  # 多边形至少需要3个点
                        points = np.array(points, dtype=np.int32)
                        cv2.fillPoly(mask, [points], 255)
                    else:
                        print(f"Not enough points to form polygon in {mask_name}: {line.strip()}")

        else:
            print(f"Mask file not found: {txt_path}")

        return mask

    def visualize(self, idx):
        """
        可视化指定索引的图像及其对应的掩码。
        :param idx: 数据集中的索引
        """
        # 加载图像
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 加载对应的掩码
        mask_name = img_name[0:-4] + '.txt'
        mask_path = os.path.join(self.masks_dir, mask_name)
        mask = self._parse_mask(mask_path, img.shape[0], img.shape[1], img_name, mask_name)

        # 显示图像和掩码
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title('Image')
        plt.imshow(img)
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title('Mask')
        plt.imshow(mask, cmap='gray')
        plt.axis('off')

        plt.tight_layout()
        plt.show()


# 使用示例
if __name__ == "__main__":
    # 创建数据集实例

    imageTrans = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    maskTrans = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    dataset = IceSegDataset(
        yaml_path='IceSeg/data.yaml',
        dataset_type='train',
        image_transforms=imageTrans,
        mask_transforms=maskTrans
    )



    # 可视化第0个样本
    dataset.visualize(765)