import matplotlib.pyplot as plt
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from modules import *
import os  # 添加用于文件夹操作的模块


def resize_and_save(image_path, output_path, target_size=(224, 224)):
    """调整图像尺寸并保存

    Args:
        image_path (str): 输入图像路径
        output_path (str): 输出保存路径
        target_size (tuple): 目标尺寸，默认(224,224)
    """
    try:
        # 打开图像并转换RGB模式
        img = Image.open(image_path).convert('RGB')

        # 调整尺寸
        resized_img = img.resize(target_size)

        # 保存图像
        resized_img.save(output_path)
        print(f"图像已保存至：{output_path}")

    except Exception as e:
        print(f"处理失败：{str(e)}")


# 修改后的可视化函数
def visualize_custom_image(model, image_path, layer_name='U_Net', img_size=224, output_folder="feature_maps"):
    # 创建输出文件夹（如果不存在）
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"创建输出文件夹：{output_folder}")

    # 图像预处理流水线
    preprocess = transforms.Compose([
        transforms.Resize((img_size, img_size)),  # 调整尺寸
        transforms.ToTensor(),  # 转为张量
        transforms.Normalize(  # 标准化（使用ImageNet统计值）
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    try:
        # 加载并预处理图像
        img = Image.open(image_path).convert('RGB')  # 确保RGB格式
        input_tensor = preprocess(img)  # 应用预处理

        input_batch = input_tensor.unsqueeze(0)  # 添加批次维度 [1, 3, H, W]
    except Exception as e:
        print(f"图像加载失败: {str(e)}")
        return

    # 注册钩子保存特征图
    feature_maps = []

    def forward_hook(module, input, output):
        feature_maps.append(output.detach().cpu())

    # 查找目标层
    target_layer = None
    for name, layer in model.named_modules():
        if name == layer_name:
            target_layer = layer
            break

    if target_layer is None:
        raise ValueError(f"找不到指定层: {layer_name}")

    # 注册钩子并前向传播
    hook = target_layer.register_forward_hook(forward_hook)
    with torch.no_grad():
        model(input_batch)
    hook.remove()

    # 可视化设置
    if len(feature_maps) == 0:
        raise RuntimeError("未捕获到特征图")

    maps = feature_maps[0]
    print(f"特征图形状: {maps.shape}")

    # 遍历每个通道并保存为单独的文件
    maps_np = maps.squeeze().numpy()
    num_channels = maps.size(1)

    for i in range(num_channels):
        channel_map = maps_np[i]
        # 归一化特征图
        channel_map = (channel_map - channel_map.min()) / (channel_map.max() - channel_map.min() + 1e-6)

        # 创建图像并保存
        plt.figure(figsize=(8, 8))
        plt.imshow(channel_map, cmap='viridis')
        plt.axis('off')
        plt.title(f'Channel {i}', fontsize=12)

        # 保存特征图
        output_path = os.path.join(output_folder, f"{layer_name}_channel_{i}.png")
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()  # 关闭当前图像以释放内存

        print(f"特征图已保存至：{output_path}")


# 使用示例
if __name__ == '__main__':
    # 初始化模型
    model = temp.NlNet(num_classes=100, in_channels=3)
    checkpoint = torch.load(r'./MiniImageNetResult/model_test.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint)
    model.eval()

    # 使用自定义图片路径
    image_path = r"E:\code\python_code\深度学习实战\CNN\特征图\原图.JPEG"

    # 可视化U-Net层特征并将每个特征图单独保存
    visualize_custom_image(
        model=model,
        image_path=image_path,
        layer_name='Upsample',
        img_size=224,
        output_folder="feature_maps"
    )
