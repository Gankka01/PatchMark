from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .patchmark_encoder import PatchGridEncoder
from .native_grid_decoder import NativeGridDecoder


class PatchMark(nn.Module):
    def __init__(self, version: int, residual_scale: float = 0.1,
                 context_size: int = 12, patch_chunk_size: int = 1024,
                 active_patch_encoding: bool = True):
        super().__init__()
        if version not in (3, 11):
            raise ValueError("QR version must be 3 or 11")
        self.version = version
        self.internal_size = 256 if version == 3 else 512
        self.grid_size = 17 + 4 * version
        self.encoder = PatchGridEncoder(
            internal_size=self.internal_size, grid_size=self.grid_size,
            residual_scale=residual_scale, active_only=active_patch_encoding,
            patch_chunk_size=patch_chunk_size)
        self.decoder = NativeGridDecoder(
            internal_size=self.internal_size, grid_size=self.grid_size,
            context_size=context_size, patch_chunk_size=patch_chunk_size)

    def encode_training(self, image, module_grid, *, active_indices=None):
        if image.ndim != 4 or image.shape[1] != 3 or image.shape[-2] != image.shape[-1]:
            raise ValueError("Expected square RGB images with shape [B,3,H,W]")
        internal = F.interpolate(image, size=(self.internal_size, self.internal_size),
                                 mode="bilinear", align_corners=False)
        residual_region = self.encoder(internal, module_grid, active_indices=active_indices)
        residual_internal = self.encoder.pad_region_to_internal(residual_region)
        residual = F.interpolate(residual_internal, size=image.shape[-2:], mode="nearest")
        return torch.clamp(image + residual, 0.0, 1.0), internal

    def encode(self, image, module_grid, *, active_indices=None):
        watermarked, internal = self.encode_training(image, module_grid, active_indices=active_indices)
        return watermarked, watermarked - image, internal

    def decode(self, image):
        internal = F.interpolate(image, size=(self.internal_size, self.internal_size),
                                 mode="bilinear", align_corners=False)
        return self.decoder(internal), internal

    def forward(self, image, module_grid):
        watermarked, _, _ = self.encode(image, module_grid)
        logits, _ = self.decode(watermarked)
        return watermarked, logits
