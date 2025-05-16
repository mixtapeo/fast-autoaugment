# -ANI FILE TO IMPORT AND CONSTRUCT YOLOV8_NANO AND INSTANTIATE IT
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import math

class ConvBNReLU(nn.Module):
    """Convolution + BatchNorm + ReLU base block"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class CSPBlock(nn.Module):
    """YOLO-inspired CSP (Cross Stage Partial) Block"""
    def __init__(self, in_channels, out_channels, num_blocks, bottleneck_ratio=0.5):
        super().__init__()
        hidden_channels = int(out_channels * bottleneck_ratio)
        
        self.conv1 = ConvBNReLU(in_channels, hidden_channels, 1)
        self.conv2 = ConvBNReLU(in_channels, hidden_channels, 1)
        self.blocks = nn.Sequential(*[
            ConvBNReLU(hidden_channels, hidden_channels, 3, padding=1)
            for _ in range(num_blocks)
        ])
        self.conv3 = ConvBNReLU(hidden_channels*2, out_channels, 1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x1 = self.blocks(x1)
        return self.conv3(torch.cat([x1, x2], dim=1))

class YOLO(nn.Module):
    def __init__(self, dataset, depth=3, num_classes=10, bottleneck_ratio=0.5):
        super().__init__()
        self.dataset = dataset
        self.in_channels = 16
        
        # Initial stem layer
        if dataset.startswith('cifar'):
            self.stem = ConvBNReLU(3, 16, kernel_size=3, stride=1, padding=1)
            layer_channels = [16, 32, 64]
        elif dataset == 'imagenet':
            self.stem = nn.Sequential(
                ConvBNReLU(3, 32, kernel_size=7, stride=2, padding=3),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
            layer_channels = [64, 128, 256, 512]
            self.in_channels = 32
        
        # Build CSP layers
        self.layers = nn.ModuleList()
        for ch in layer_channels:
            self.layers.append(
                CSPBlock(self.in_channels, ch, num_blocks=depth, 
                        bottleneck_ratio=bottleneck_ratio)
            )
            self.in_channels = ch
        
        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.in_channels, num_classes)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.stem(x)
        for layer in self.layers:
            x = layer(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)