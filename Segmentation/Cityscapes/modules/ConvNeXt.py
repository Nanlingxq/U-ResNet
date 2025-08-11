# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from torchvision.models import convnext_tiny


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
                 drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3],
                 pretrained=None):  # 添加预训练参数
        super().__init__()
        self.out_indices = out_indices
        self.pretrained = pretrained
        self.dims = dims
        # 修改下采样层适应512x1024输入
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
        total_depth = sum(depths)
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                  layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm_layers = nn.ModuleList()
        for i in range(4):
            norm_layer = LayerNorm(self.dims[i], eps=1e-6, data_format="channels_first")
            self.norm_layers.append(norm_layer)

        # 初始化权重
        self.apply(self._init_weights)
        
        # 加载预训练权重
        if pretrained:
            self.load_pretrained(pretrained)

    def load_pretrained(self, pretrained):
        if pretrained.startswith('http'):
            checkpoint = torch.hub.load_state_dict_from_url(pretrained, map_location='cpu')
        else:
            checkpoint = torch.load(pretrained, map_location='cpu')
        
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # 过滤分类头权重
        state_dict = {k: v for k, v in state_dict.items() 
                     if not k.startswith('head.') and not k.startswith('norm.')}
        
        # 加载权重
        msg = self.load_state_dict(state_dict, strict=False)
        print(f'Loaded pretrained weights: {msg}')
        # 归一化层
        

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

        def get_norm_layer(channels):
            groups = min(32, channels)
            if channels % groups != 0:
                groups = 1  # fallback to LayerNorm-like behavior
            return nn.GroupNorm(groups, channels)

        # PSP模块
        self.psp_modules = nn.ModuleList()
        for scale in pool_scales:
            self.psp_modules.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(in_channels[-1], channels, kernel_size=1, bias=False),
                get_norm_layer(channels),
                nn.ReLU(inplace=False)
            ))

        self.psp_bottleneck = nn.Sequential(
            nn.Conv2d(in_channels[-1] + len(pool_scales) * channels, channels * 2, kernel_size=3, padding=1, bias=False),
            get_norm_layer(channels * 2),
            nn.ReLU(inplace=False),
            nn.Dropout2d(0.2),
            nn.Conv2d(channels * 2, channels, kernel_size=1)
        )

        # FPN模块
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for in_channel in in_channels[:-1]:
            self.lateral_convs.append(nn.Sequential(
                nn.Conv2d(in_channel, channels * 2, kernel_size=1, bias=False),
                get_norm_layer(channels * 2),
                nn.ReLU(inplace=False),
                nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1, bias=False),
                get_norm_layer(channels),
                nn.ReLU(inplace=False)
            ))

            self.fpn_convs.append(nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
                get_norm_layer(channels),
                nn.ReLU(inplace=False)
            ))

        self.fpn_bottleneck = nn.Sequential(
            nn.Conv2d(len(in_channels) * channels, channels * 2, kernel_size=3, padding=1, bias=False),
            get_norm_layer(channels * 2),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1, bias=False),
            get_norm_layer(channels),
            nn.ReLU(inplace=False)
        )

        self.cls_seg = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            get_norm_layer(channels),
            nn.ReLU(inplace=False),
            nn.Dropout2d(0.2),
            nn.Conv2d(channels, num_classes, kernel_size=1)
        )

        self.upsample4x = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

    def forward(self, inputs):
        psp_outs = [inputs[-1]]
        for psp_module in self.psp_modules:
            psp_out = psp_module(inputs[-1])
            psp_out = F.interpolate(psp_out, size=inputs[-1].shape[2:], mode='bilinear', align_corners=False)
            psp_outs.append(psp_out)

        psp_out = torch.cat(psp_outs, dim=1)
        psp_out = self.psp_bottleneck(psp_out)

        laterals = [conv(inputs[i]) for i, conv in enumerate(self.lateral_convs)]
        laterals.append(psp_out)

        for i in range(len(laterals) - 1, 0, -1):
            t = F.interpolate(
                laterals[i],
                size=laterals[i - 1].shape[2:],
                mode='bilinear',
                align_corners=False
            )

            laterals[i - 1] = t + laterals[i - 1]

        fpn_outs = [self.fpn_convs[i](laterals[i]) for i in range(len(laterals) - 1)]
        fpn_outs.append(laterals[-1])

        target_size = fpn_outs[0].shape[2:]
        fpn_outs = [F.interpolate(f, target_size, mode='bilinear', align_corners=False)
                    if f.shape[2:] != target_size else f
                    for f in fpn_outs]

        fpn_out = torch.cat(fpn_outs, dim=1)
        fpn_out = self.fpn_bottleneck(fpn_out)

        output = self.cls_seg(fpn_out)
        return self.upsample4x(output)

class ConvNeXtUPerNet(nn.Module):
    def __init__(self, num_classes, backbone='convnext_small', in_chans=3, 
                 pretrained_backbone=None, with_aux_head=True):  # 默认添加辅助头
        super().__init__()
        
        # 骨干网络配置
        backbone_config = {
            'convnext_tiny': {'depths': [3, 3, 9, 3], 'dims': [96, 192, 384, 768]},
            'convnext_small': {'depths': [3, 3, 27, 3], 'dims': [96, 192, 384, 768]},
            'convnext_base': {'depths': [3, 3, 27, 3], 'dims': [128, 256, 512, 1024]},
            'convnext_large': {'depths': [3, 3, 27, 3], 'dims': [192, 384, 768, 1536]}
        }
        
        self.with_aux_head = with_aux_head
        if backbone not in backbone_config:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        config = backbone_config[backbone]
        self.backbone = ConvNeXtBackbone(
            in_chans=in_chans,
            depths=config['depths'],
            dims=config['dims'],
            pretrained=pretrained_backbone  # 添加预训练支持
        )
        
        # 分割头
        self.decode_head = UPerHead(
            in_channels=config['dims'],
            channels=512,
            num_classes=num_classes
        )
        
        # 辅助头（默认启用）
        if with_aux_head:
            self.auxiliary_head = nn.Sequential(
                nn.Conv2d(config['dims'][2], 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=False),
                nn.Dropout2d(0.1),
                nn.Conv2d(256, num_classes, kernel_size=1),
                nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
            )
        else:
            self.auxiliary_head = None

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
        # seg_out = F.interpolate(
        #     seg_out, 
        #     scale_factor=4, 
        #     mode='bilinear', 
        #     align_corners=False)
        
        outputs = seg_out
        
        # 训练时添加辅助输出
        if self.training and self.with_aux_head:
            outputs = [outputs]
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

def create_cityscapes_model():
    return ConvNeXtUPerNet(
        num_classes=19,  # Cityscapes有19个类别
        backbone='convnext_small',
        with_aux_head=True
    )