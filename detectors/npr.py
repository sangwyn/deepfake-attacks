"""NPR detector architecture.

This is a compact implementation of the two-stage residual ResNet published in
``chuangchuangtan/NPR-DeepfakeDetection``.  Attribute names intentionally match
the released checkpoint.
"""

import torch.nn as nn
import torch.nn.functional as F


def _conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)


def _conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)


class _Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = _conv1x1(in_channels, channels)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = _conv3x3(channels, channels, stride)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv3 = _conv1x1(channels, channels * self.expansion)
        self.bn3 = nn.BatchNorm2d(channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        return self.relu(x + identity)


class NPRResNet(nn.Module):
    """NPR's truncated ResNet-50 operating on nearest-neighbour residuals."""

    def __init__(self, num_classes=1):
        super().__init__()
        self.unfoldSize = 2
        self.unfoldIndex = 0
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = self._make_layer(64, 3)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, num_classes)

    def _make_layer(self, channels, blocks, stride=1):
        out_channels = channels * _Bottleneck.expansion
        downsample = None
        if stride != 1 or self.inplanes != out_channels:
            downsample = nn.Sequential(
                _conv1x1(self.inplanes, out_channels, stride),
                nn.BatchNorm2d(out_channels),
            )
        layers = [_Bottleneck(self.inplanes, channels, stride, downsample)]
        self.inplanes = out_channels
        layers.extend(_Bottleneck(self.inplanes, channels) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    @staticmethod
    def _residual(x):
        down = F.interpolate(
            x, scale_factor=0.5, mode="nearest", recompute_scale_factor=True
        )
        up = F.interpolate(
            down, scale_factor=2.0, mode="nearest", recompute_scale_factor=True
        )
        return x - up

    def forward(self, x):
        x = self._residual(x) * (2.0 / 3.0)
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer2(self.layer1(x))
        return self.fc1(self.avgpool(x).flatten(1))


def resnet50(num_classes=1):
    return NPRResNet(num_classes=num_classes)
