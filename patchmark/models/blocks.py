from __future__ import annotations
import torch
from torch import nn

class LayerNorm2d(nn.Module):

    def __init__(self, channels: int, eps: float=1e-06):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

class SqueezeExcitation(nn.Module):

    def __init__(self, channels: int, reduction: int=16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.net = nn.Sequential(nn.Conv2d(channels, hidden, 1), nn.GELU(), nn.Conv2d(hidden, channels, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(self.pool(x))

class ConvNormAct(nn.Sequential):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int=3, stride: int=1, padding: int | None=None, groups: int=1, norm: bool=True, activation: bool=True):
        if padding is None:
            padding = kernel_size // 2
        layers: list[nn.Module] = [nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups, bias=not norm)]
        if norm:
            layers.append(LayerNorm2d(out_channels))
        if activation:
            layers.append(nn.GELU())
        super().__init__(*layers)

class ResidualConvBlock(nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(ConvNormAct(channels, channels, 3), ConvNormAct(channels, channels, 3, activation=False))
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))
