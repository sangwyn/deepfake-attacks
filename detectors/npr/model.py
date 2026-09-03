"""NPR detector architecture reproduced from the verified team adapter."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, 3, stride, 1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, 1, stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1, self.bn1 = conv1x1(inplanes, planes), nn.BatchNorm2d(planes)
        self.conv2, self.bn2 = conv3x3(planes, planes, stride), nn.BatchNorm2d(planes)
        self.conv3, self.bn3 = conv1x1(planes, planes * 4), nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample, self.stride = downsample, stride

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + identity)


class NPRResNet(nn.Module):
    def __init__(self, layers=(3, 4, 6, 3), num_classes=1):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, 2, 1, bias=False)
        self.bn1, self.relu = nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, num_classes)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * 4:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * 4, stride),
                                       nn.BatchNorm2d(planes * 4))
        layers = [Bottleneck(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * 4
        layers.extend(Bottleneck(self.inplanes, planes) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    @staticmethod
    def _npr(x):
        reduced = F.interpolate(
            F.interpolate(x, scale_factor=0.5, mode="nearest",
                          recompute_scale_factor=True),
            scale_factor=2.0, mode="nearest", recompute_scale_factor=True
        )
        return x - reduced

    def forward(self, x):
        x = self._npr(x) * (2.0 / 3.0)
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer2(self.layer1(x))
        return self.fc1(self.avgpool(x).flatten(1))
