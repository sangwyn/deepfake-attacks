"""Differentiable test-time preprocessing used by AIDE."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _dct_matrix(size):
    frequency = torch.arange(size, dtype=torch.float32).unsqueeze(1)
    position = torch.arange(size, dtype=torch.float32).unsqueeze(0) + 0.5
    matrix = torch.cos(torch.pi * frequency * position / size)
    matrix[0] *= math.sqrt(1.0 / size)
    matrix[1:] *= math.sqrt(2.0 / size)
    return matrix


def _band_mask(size, start, end):
    row = torch.arange(size).unsqueeze(1)
    column = torch.arange(size).unsqueeze(0)
    return ((row + column >= start) & (row + column <= end)).float()


class DCTPatchSelector(nn.Module):
    """Select the two lowest- and highest-energy overlapping DCT patches."""

    def __init__(self, window_size=32, stride=16, grades=6):
        super().__init__()
        self.window_size = window_size
        self.grades = grades
        matrix = _dct_matrix(window_size)
        self.register_buffer("dct", matrix, persistent=False)
        masks = []
        counts = []
        for index in range(grades):
            start = window_size * 2.0 / grades * index
            end = window_size * 2.0 / grades * (index + 1)
            mask = _band_mask(window_size, start, end)
            masks.append(mask)
            counts.append(mask.sum())
        self.register_buffer("grade_masks", torch.stack(masks), persistent=False)
        self.register_buffer("grade_counts", torch.stack(counts), persistent=False)
        self.unfold = nn.Unfold(window_size, stride=stride)

    def forward(self, image):
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError("AIDE preprocessing expects a [3, H, W] tensor")
        patches = self.unfold(image.unsqueeze(0)).squeeze(0).transpose(0, 1)
        if patches.shape[0] < 2:
            raise ValueError(
                "AIDE requires an image large enough for two 32x32 patches"
            )
        patches = patches.reshape(-1, 3, self.window_size, self.window_size)
        coefficients = self.dct @ patches @ self.dct.t()
        magnitude = torch.log(coefficients.abs() + 1.0)
        band_energy = (magnitude.unsqueeze(1) * self.grade_masks[None, :, None]).sum(
            dim=(2, 3, 4)
        ) / self.grade_counts[None]
        weights = magnitude.new_tensor([2.0**index for index in range(self.grades)])
        scores = (band_energy * weights).sum(dim=1)
        order = scores.argsort()
        return (
            patches[order[0]],
            patches[order[-1]],
            patches[order[1]],
            patches[order[-2]],
        )


class AIDEPreprocessor(nn.Module):
    """Map ``[N, 3, H, W]`` RGB images in ``[0,1]`` to AIDE's five views."""

    def __init__(self):
        super().__init__()
        self.selector = DCTPatchSelector()

    @staticmethod
    def _resize_normalize(image):
        image = F.interpolate(
            image.unsqueeze(0),
            size=(256, 256),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        ).squeeze(0)
        mean = image.new_tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = image.new_tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return (image - mean) / std

    def forward(self, images):
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("AIDE input must have shape [N, 3, H, W]")
        batches = []
        for image in images:
            minimum, maximum, second_minimum, second_maximum = self.selector(image)
            views = (minimum, maximum, second_minimum, second_maximum, image)
            batches.append(torch.stack([self._resize_normalize(x) for x in views]))
        return torch.stack(batches)
