# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model


class Block(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = input + self.drop_path(x)
        return x

class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., 
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class ConvNeXtBackbone(nn.Module):
    def __init__(self, in_chans=3, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], 
                 drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3]):
        super().__init__()
        self.out_indices = out_indices
        
        # 下采样层
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        # 阶段块
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                  layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        # 归一化层
        self.norm_layers = nn.ModuleList()
        for i in range(4):
            norm_layer = LayerNorm(dims[i], eps=1e-6, data_format="channels_first")
            self.norm_layers.append(norm_layer)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        outputs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i in self.out_indices:
                x = self.norm_layers[i](x)
                outputs.append(x)
        return outputs

class UPerHead(nn.Module):
    def __init__(self, in_channels, channels, num_classes, pool_scales=(1, 2, 3, 6)):
        super().__init__()
        # PSP模块 - 金字塔池化
        self.psp_modules = nn.ModuleList()
        for scale in pool_scales:
            self.psp_modules.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(in_channels[-1], channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=False)
            ))
        
        self.psp_bottleneck = nn.Sequential(
            nn.Conv2d(in_channels[-1] + len(pool_scales) * channels, channels, 
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=False),
            nn.Dropout2d(0.1)
        )
        
        # FPN模块 - 特征金字塔网络
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        
        for in_channel in in_channels[:-1]:  # 跳过最后一层
            self.lateral_convs.append(nn.Sequential(
                nn.Conv2d(in_channel, channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=False))
            )
            
            self.fpn_convs.append(nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=False))
            )
        
        # FPN瓶颈层
        self.fpn_bottleneck = nn.Sequential(
            nn.Conv2d(len(in_channels) * channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=False)
        )
        
        # 分类头
        self.cls_seg = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=False),
            nn.Dropout2d(0.1),
            nn.Conv2d(channels, num_classes, kernel_size=1)
        )

    def forward(self, inputs):
        # 输入应该是4个特征图: [stage1, stage2, stage3, stage4]
        assert len(inputs) == 4
        
        # 1. 构建PSP金字塔池化
        psp_outs = [inputs[-1]]
        for psp_module in self.psp_modules:
            resized = F.interpolate(
                psp_module(inputs[-1]),
                size=inputs[-1].shape[2:],
                mode='bilinear',
                align_corners=False)
            psp_outs.append(resized)
        
        psp_out = torch.cat(psp_outs, dim=1)
        psp_out = self.psp_bottleneck(psp_out)
        
        # 2. 构建FPN特征金字塔
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        laterals.append(psp_out)  # 添加PSP输出作为顶层特征
        
        # 自顶向下路径
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            interpolated = F.interpolate(
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=False
            )
            # 使用非原地操作替代 +=
            laterals[i - 1] = laterals[i - 1] + interpolated
        
        # 3. 构建输出特征
        fpn_outs = []
        for i in range(used_backbone_levels - 1):
            fpn_out = self.fpn_convs[i](laterals[i])
            fpn_outs.append(fpn_out)
        fpn_outs.append(laterals[-1])
        
        # 上采样到相同尺寸
        target_size = fpn_outs[0].shape[2:]
        for i in range(len(fpn_outs)):
            if fpn_outs[i].shape[2:] != target_size:
                fpn_outs[i] = F.interpolate(
                    fpn_outs[i], 
                    size=target_size, 
                    mode='bilinear',
                    align_corners=False)
        
        # 合并特征
        fpn_out = torch.cat(fpn_outs, dim=1)
        fpn_out = self.fpn_bottleneck(fpn_out)
        
        # 4. 分类
        output = self.cls_seg(fpn_out)
        return output

class ConvNeXtUPerNet(nn.Module):
    def __init__(self, num_classes, backbone='convnext_base', in_chans=3, img_size=640, with_aux_head=False):
        super().__init__()
        
        # 创建骨干网络
        if backbone == 'convnext_tiny':
            self.backbone = ConvNeXtBackbone(
                in_chans=in_chans,
                depths=[3, 3, 9, 3],
                dims=[96, 192, 384, 768]
            )
            in_channels = [96, 192, 384, 768]
        elif backbone == 'convnext_small':
            self.backbone = ConvNeXtBackbone(
                in_chans=in_chans,
                depths=[3, 3, 27, 3],
                dims=[96, 192, 384, 768]
            )
            in_channels = [96, 192, 384, 768]
        elif backbone == 'convnext_base':
            self.backbone = ConvNeXtBackbone(
                in_chans=in_chans,
                depths=[3, 3, 27, 3],
                dims=[128, 256, 512, 1024]
            )
            in_channels = [128, 256, 512, 1024]
        elif backbone == 'convnext_large':
            self.backbone = ConvNeXtBackbone(
                in_chans=in_chans,
                depths=[3, 3, 27, 3],
                dims=[192, 384, 768, 1536]
            )
            in_channels = [192, 384, 768, 1536]
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # 创建分割头
        self.decode_head = UPerHead(
            in_channels=in_channels,
            channels=512,
            num_classes=num_classes,
            pool_scales=(1, 2, 3, 6)
        )
        
        # 可选的辅助头
        if with_aux_head:
            self.auxiliary_head = nn.Sequential(
                nn.Conv2d(in_channels[2], 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=False),
                nn.Dropout2d(0.1),
                nn.Conv2d(256, num_classes, kernel_size=1),
                # 添加上采样层
                nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
            )
        else:
            self.auxiliary_head = None
        
        # 初始化权重
        self.apply(self._init_weights)
        self.name = "ConvNeXt"
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 骨干网络提取特征
        features = self.backbone(x)
        
        # 主分割头
        seg_out = self.decode_head(features)
        
        # 上采样到原始尺寸
        seg_out = F.interpolate(
            seg_out, 
            scale_factor=4, 
            mode='bilinear', 
            align_corners=False)
        
        outputs = [seg_out]
        
        # 训练时添加辅助输出
        if self.training:
            aux_out = self.auxiliary_head(features[2])
            if aux_out.size()[-2:] != seg_out.size()[-2:]:
                aux_out = F.interpolate(
                    aux_out, 
                    size=seg_out.shape[-2:],  # 匹配主输出尺寸
                    mode='bilinear', 
                    align_corners=False
                )
            outputs.append(aux_out)
        
        return outputs

model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_tiny_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth",
    "convnext_small_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth",
    "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
}

@register_model
def convnext_tiny(pretrained=False,in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_tiny_22k'] if in_22k else model_urls['convnext_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_small(pretrained=False,in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_small_22k'] if in_22k else model_urls['convnext_small_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_base(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    if pretrained:
        url = model_urls['convnext_base_22k'] if in_22k else model_urls['convnext_base_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_large(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    if pretrained:
        url = model_urls['convnext_large_22k'] if in_22k else model_urls['convnext_large_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_xlarge(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)
    if pretrained:
        assert in_22k, "only ImageNet-22K pre-trained ConvNeXt-XL is available; please set in_22k=True"
        url = model_urls['convnext_xlarge_22k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

