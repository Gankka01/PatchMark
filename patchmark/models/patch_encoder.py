from __future__ import annotations
import torch
from torch import nn
from .blocks import SqueezeExcitation

class SharedPatchEncoder(nn.Module):
    patch_size = 8

    def __init__(self, out_channels: int=3):
        super().__init__()
        self.enc1 = self._conv_block(3, 64)
        self.norm1_e = nn.LayerNorm([64, 8, 8])
        self.down1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bottleneck1 = self._conv_block(128, 256)
        self.norm_b1 = nn.LayerNorm([256, 4, 4])
        self.bottleneck2 = self._conv_block(256, 512)
        self.norm_b2 = nn.LayerNorm([512, 4, 4])
        self.se_b2 = SqueezeExcitation(512, reduction=16)
        self.reduce_512_256 = nn.Sequential(nn.Conv2d(512, 256, 1), nn.GELU())
        self.bottleneck3 = self._conv_block(256, 128)
        self.norm_b3 = nn.LayerNorm([128, 4, 4])
        self.upconv1 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(128 + 64, 128)
        self.norm1_d = nn.LayerNorm([128, 8, 8])
        self.dec2 = self._conv_block(128, 64)
        self.norm2_d = nn.LayerNorm([64, 8, 8])
        self.final_conv = nn.Conv2d(64, out_channels, 1)

    @staticmethod
    def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.GELU(), nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.GELU())

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        if patches.ndim != 4 or patches.shape[1:] != (3, 8, 8):
            raise ValueError(f'Expected [N,3,8,8], got {tuple(patches.shape)}')
        e1 = self.norm1_e(self.enc1(patches))
        x = self.down1(e1)
        x = self.norm_b1(self.bottleneck1(x))
        x = self.norm_b2(self.bottleneck2(x))
        x = self.se_b2(x)
        x = self.reduce_512_256(x)
        x = self.norm_b3(self.bottleneck3(x))
        x = self.upconv1(x)
        x = torch.cat([x, e1], dim=1)
        x = self.norm1_d(self.dec1(x))
        x = self.norm2_d(self.dec2(x))
        return torch.tanh(self.final_conv(x))
