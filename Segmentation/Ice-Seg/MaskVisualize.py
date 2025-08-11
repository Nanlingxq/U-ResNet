import torch
import os
import numpy as np
from torchvision import transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn
from modules import *

# 定义反归一化函数
def reverse_normalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = tensor * std + mean
    tensor = torch.clamp(tensor, 0, 1)
    return tensor.permute(1, 2, 0).numpy()

# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = U_ResNet.UResNet(seg_classes=1, in_channels=3, base_c=64).to(device)
model.load_state_dict(torch.load('Result/U-ResNet(64)_best_weights.pth', map_location=device))
model.eval()

# 数据预处理
image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 创建保存结果的文件夹
save_dir = "Mask"
os.makedirs(save_dir, exist_ok=True)

# 修改后的可视化函数 - 保存结果到文件
def save_single_sample(image_path, model, device, save_path):
    """
    参数：
        image_path (str): 图像文件路径
        model (nn.Module): 加载的模型
        device (torch.device): 计算设备
        save_path (str): 结果保存路径
    """
    # 读取图像
    image = Image.open(image_path).convert('RGB')
    image = image_transform(image).unsqueeze(0).to(device)  # 添加批次维度

    with torch.no_grad():
        output = model(image)
        pred_mask = (torch.sigmoid(output) > 0.5).float()

    # 转换数据为可显示格式
    img_np = reverse_normalize(image.cpu().squeeze(0))
    pred_mask_np = pred_mask.squeeze().cpu().numpy()

    # 创建图形
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # 绘制结果
    axes[0].imshow(img_np)
    axes[0].set_title('Input Image')
    axes[0].axis('off')

    axes[1].imshow(pred_mask_np, cmap='gray')
    axes[1].set_title('Predicted Mask')
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)  # 关闭图形以释放内存

# 从指定文件夹中读取图像数据
image_folder = "TestImage/images"  # 替换为你的图像文件夹路径
image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

# 处理前50张图片
num_samples = min(300, len(image_files))

for i in range(num_samples):
    image_path = image_files[i]
    save_path = os.path.join(save_dir, f"result_{i:03d}.png")
    save_single_sample(image_path, model, device, save_path)
    print(f"Saved result for sample {i} to {save_path}")

print(f"Successfully processed and saved {num_samples} images to {save_dir} directory.")